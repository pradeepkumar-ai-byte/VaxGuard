from enum import Enum
from pydantic import BaseModel, Field

class AttackCategory(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    DATA_EXTRACTION = "data_extraction"
    TOKEN_SMUGGLING = "token_smuggling"
    SEMANTIC_MANIPULATION = "semantic_manipulation"
    CHAIN_POISONING = "chain_poisoning"

class AttackVector(BaseModel):
    id: str
    name: str
    category: AttackCategory
    payload: str
    expected_defense: str
    severity: int = Field(ge=1, le=10)
    description: str
