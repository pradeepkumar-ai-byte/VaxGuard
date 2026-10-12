"""Configuration models and serialization for NVIDIA NeMo Guardrails compatibility.

Supports reading and writing enterprise RailsConfig in YAML and Colang (.co) formats,
with zero external heavy binary dependencies.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, Field

from vaxguard.nemo.colang import ColangDocument, ColangParser, FlowManager


class ModelConfig(BaseModel):
    """Configuration for an underlying language or embedding model."""

    type: str = "main"
    engine: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    parameters: Dict[str, Any] = Field(default_factory=dict)


class RailsBranchConfig(BaseModel):
    """Specification of active flows for a specific rail phase (input, output, dialog)."""

    flows: List[str] = Field(default_factory=list)


class RailsDefinitionConfig(BaseModel):
    """Grouping of active input, output, and dialog rail flows."""

    input: RailsBranchConfig = Field(default_factory=RailsBranchConfig)
    output: RailsBranchConfig = Field(default_factory=RailsBranchConfig)
    dialog: RailsBranchConfig = Field(default_factory=RailsBranchConfig)


class PromptConfig(BaseModel):
    """Task-specific prompt template used by NeMo Guardrails."""

    task: str
    content: str
    models: List[str] = Field(default_factory=list)


class InstructionConfig(BaseModel):
    """System-level instructions defining assistant posture."""

    type: str = "general"
    content: str


class RailsConfig(BaseModel):
    """Full enterprise Rails configuration conforming to NVIDIA NeMo Guardrails."""

    models: List[ModelConfig] = Field(default_factory=list)
    rails: RailsDefinitionConfig = Field(default_factory=RailsDefinitionConfig)
    prompts: List[PromptConfig] = Field(default_factory=list)
    instructions: List[InstructionConfig] = Field(default_factory=list)
    colang_files: List[str] = Field(default_factory=list)
    colang_docs: List[ColangDocument] = Field(default_factory=list, exclude=True)
    custom_data: Dict[str, Any] = Field(default_factory=dict)

    def get_primary_model(self) -> Optional[ModelConfig]:
        """Return the primary 'main' generation model, or first configured model."""
        for model in self.models:
            if model.type == "main":
                return model
        return self.models[0] if self.models else None

    def get_prompt(self, task_name: str) -> Optional[str]:
        """Lookup task-specific prompt content by name."""
        for p in self.prompts:
            if p.task == task_name:
                return p.content
        return None

    def get_combined_colang_document(self) -> ColangDocument:
        """Combine all loaded Colang documents into a single document."""
        combined = ColangDocument()
        for doc in self.colang_docs:
            combined = combined.merge(doc)
        return combined

    def create_flow_manager(self) -> FlowManager:
        """Instantiate a FlowManager pre-loaded with all registered rails and flows."""
        combined_doc = self.get_combined_colang_document()
        return FlowManager(document=combined_doc)

    def to_yaml(self) -> str:
        """Serialize configuration into standard NeMo Guardrails YAML format."""
        data: Dict[str, Any] = {
            "models": [m.model_dump(exclude_none=True) for m in self.models],
            "rails": {
                "input": {"flows": self.rails.input.flows},
                "output": {"flows": self.rails.output.flows},
                "dialog": {"flows": self.rails.dialog.flows},
            },
            "prompts": [p.model_dump(exclude_none=True) for p in self.prompts],
            "instructions": [i.model_dump(exclude_none=True) for i in self.instructions],
        }
        if self.custom_data:
            data["custom_data"] = self.custom_data
        return yaml.dump(data, sort_keys=False, default_flow_style=False)

    def to_colang(self) -> str:
        """Serialize combined Colang definitions to Colang 1.0 code."""
        return self.get_combined_colang_document().to_colang()

    def save(
        self,
        dir_path: Union[str, Path],
        config_filename: str = "config.yml",
        colang_filename: str = "rails.co",
    ) -> None:
        """Save this configuration and its Colang definitions to a directory."""
        target_dir = Path(dir_path)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Write YAML config
        yaml_content = self.to_yaml()
        (target_dir / config_filename).write_text(yaml_content, encoding="utf-8")

        # Write combined Colang script if any flows/intents exist
        colang_content = self.to_colang()
        if colang_content.strip():
            (target_dir / colang_filename).write_text(colang_content, encoding="utf-8")

    @classmethod
    def from_yaml_str(
        cls,
        yaml_content: str,
        colang_content: Optional[str] = None,
    ) -> RailsConfig:
        """Construct RailsConfig from YAML text and optional Colang string."""
        raw_data = yaml.safe_load(yaml_content) or {}
        raw_models = raw_data.get("models", [])
        raw_rails = raw_data.get("rails", {})
        raw_prompts = raw_data.get("prompts", [])
        raw_instructions = raw_data.get("instructions", [])

        models = [ModelConfig(**m) for m in raw_models]
        rails = RailsDefinitionConfig(
            input=RailsBranchConfig(flows=raw_rails.get("input", {}).get("flows", [])),
            output=RailsBranchConfig(flows=raw_rails.get("output", {}).get("flows", [])),
            dialog=RailsBranchConfig(flows=raw_rails.get("dialog", {}).get("flows", [])),
        )
        prompts = [PromptConfig(**p) for p in raw_prompts]
        instructions = [InstructionConfig(**i) for i in raw_instructions]

        instance = cls(
            models=models,
            rails=rails,
            prompts=prompts,
            instructions=instructions,
            custom_data={
                k: v
                for k, v in raw_data.items()
                if k not in ("models", "rails", "prompts", "instructions")
            },
        )

        if colang_content:
            parser = ColangParser()
            doc = parser.parse_content(colang_content)
            instance.colang_docs.append(doc)

        return instance

    @classmethod
    def from_path(cls, path: Union[str, Path]) -> RailsConfig:
        """Load RailsConfig and all associated .co Colang files from a directory or file.

        Args:
            path: Path to a directory containing config.yml and .co files,
                  or direct path to a YAML/Colang file.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Path does not exist: {p}")

        parser = ColangParser()

        if p.is_dir():
            # Look for config.yml or config.yaml
            config_file = p / "config.yml"
            if not config_file.exists():
                config_file = p / "config.yaml"

            if config_file.exists():
                content = config_file.read_text(encoding="utf-8")
                config = cls.from_yaml_str(content)
            else:
                config = cls()

            # Find and parse all .co files in directory
            for co_path in sorted(p.glob("**/*.co")):
                doc = parser.parse_file(co_path)
                config.colang_docs.append(doc)
                config.colang_files.append(str(co_path))

            return config

        # If it's a file
        if p.suffix in (".yml", ".yaml"):
            content = p.read_text(encoding="utf-8")
            config = cls.from_yaml_str(content)

            # Check if there are .co files in the parent directory
            for co_path in sorted(p.parent.glob("*.co")):
                doc = parser.parse_file(co_path)
                config.colang_docs.append(doc)
                config.colang_files.append(str(co_path))
            return config

        if p.suffix == ".co":
            config = cls()
            doc = parser.parse_file(p)
            config.colang_docs.append(doc)
            config.colang_files.append(str(p))
            return config

        raise ValueError(f"Unsupported file format for RailsConfig: {p}")
