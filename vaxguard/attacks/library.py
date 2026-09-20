import os
import yaml
from typing import List
from vaxguard.models.attack import AttackVector, AttackCategory

class AttackLibrary:
    """
    Loads and serves attack vectors from the YAML taxonomy.
    Validates all data against Pydantic models on load.
    """
    def __init__(self, yaml_path: str = None):
        if yaml_path is None:
            # Default to data/attacks.yaml relative to project root
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            yaml_path = os.path.join(project_root, "data", "attacks.yaml")
            
        self.yaml_path = yaml_path
        self.attacks: List[AttackVector] = []
        self._load_attacks()

    def _load_attacks(self):
        with open(self.yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or []
        
        # Pydantic validates each item as it is loaded
        self.attacks = [AttackVector(**item) for item in data]

    def get_all(self) -> List[AttackVector]:
        return self.attacks

    def get_by_category(self, category: AttackCategory) -> List[AttackVector]:
        return [attack for attack in self.attacks if attack.category == category]

    def get_by_id(self, attack_id: str) -> AttackVector:
        for attack in self.attacks:
            if attack.id == attack_id:
                return attack
        return None

