"""
vocabulary.py

Vocabulary loader that reads from the vocabulary.txt file and provides
token mapping functionality for the action-level model.

Author: Gordon Doore
Date Created: 2025-08-22
"""

import os
from typing import Dict, List

def find_project_root() -> str:
    """Find the project root directory by looking for specific marker files"""
    current_dir = os.path.abspath(os.path.dirname(__file__))
    
    # Look for project markers (files that should be in the root)
    markers = ['README.md', 'requirements.txt', '.git']
    
    while current_dir != os.path.dirname(current_dir):  # Not at filesystem root
        if any(os.path.exists(os.path.join(current_dir, marker)) for marker in markers):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    
    # Fallback: assume we're in a known structure
    # This file is at src/specgen/vocabulary.py, so project root is ../..
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

class ActionVocabulary:
    """Vocabulary class that loads from the vocabulary.txt file"""
    
    def __init__(self, vocab_file: str = None):
        if vocab_file is None:
            # Find project root and look for vocabulary file in multiple locations
            project_root = find_project_root()
            possible_paths = [
                os.path.join(project_root, 'src', 'specgen', 'vocabulary.txt'),
                os.path.join(project_root, 'docs', 'vocabulary.txt'),
                os.path.join(os.path.dirname(__file__), 'vocabulary.txt'),
            ]
            
            # Use the first path that exists
            vocab_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    vocab_file = path
                    break
            
            if vocab_file is None:
                raise FileNotFoundError(f"Could not find vocabulary.txt in any of these locations: {possible_paths}")
        
        self.vocab_file = vocab_file
        self.vocab = self._load_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
    
    def _load_vocab(self) -> List[str]:
        """Load vocabulary from the vocabulary.txt file"""
        vocab = []
        
        with open(self.vocab_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('%') or line.startswith('$'):
                    continue
                
                # Extract token from quotes
                if line.startswith('"'):
                    # Handle various quote formats: "token", "token," "token"
                    token = line.rstrip(',').strip('"')
                    # Skip empty tokens
                    if token:
                        vocab.append(token)
        
        return vocab
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        return [self.token_to_id.get(token, self.token_to_id.get("<UNK>", 0)) for token in tokens]
    
    def decode(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens"""
        return [self.id_to_token.get(id, "<UNK>") for id in ids]
    
    def size(self) -> int:
        """Get vocabulary size"""
        return len(self.vocab)
    
    def get_token_sets(self) -> Dict[str, set]:
        """Get categorized token sets for grammar enforcement"""
        
        # Actions (operations)
        actions = {token for token in self.vocab if token.startswith("ADD_") or token in {"CONNECT", "APPEND_TO_BAY", "VALIDATE", "EMIT_SPEC"}}
        
        # Parameters
        params = {token for token in self.vocab if token.endswith("=")}
        
        # Enum values for different categories
        bay_kinds = {"LINE", "TRANSFORMER", "FEEDER", "SHUNT", "COUPLER", "GENERATOR"}
        breaker_kinds = {"SF6", "VACUUM", "OIL", "AIRBLAST"}
        disconnector_kinds = {"CENTER_BREAK", "DOUBLE_BREAK", "PANTOGRAPH", "EARTH_SWITCH_COMBINED"}
        transformer_kinds = {"TWO_WINDING", "AUTO", "THREE_WINDING", "GROUNDING"}
        line_kinds = {"OHL", "UGC"}
        
        # IDs by component type
        bus_ids = {token for token in self.vocab if token.startswith("bus")}
        breaker_ids = {token for token in self.vocab if token.startswith("breaker")}
        coupler_ids = {token for token in self.vocab if token.startswith("coupler")}
        bay_ids = {token for token in self.vocab if token.startswith("bay")}
        disconnector_ids = {token for token in self.vocab if token.startswith("disconnector")}
        transformer_ids = {token for token in self.vocab if token.startswith("transformer")}
        line_ids = {token for token in self.vocab if token.startswith("line")}
        
        # Voltage and MVA values
        voltages = {token for token in self.vocab if token.endswith("kv")}
        mva_values = {token for token in self.vocab if token.endswith("mva")}
        
        # New token categories
        percentz_values = {token for token in self.vocab if token.endswith("percentZ")}
        thermal_values = {token for token in self.vocab if token.endswith("A") and not token.startswith("Y")}
        ka_values = {token for token in self.vocab if token.endswith("kA")}
        counts = {token for token in self.vocab if token.isdigit()}
        reliability_targets = {"N_0", "N_1", "N_2", "REDUNDANT", "MINIMAL"}
        vector_groups = {token for token in self.vocab if any(token.startswith(prefix) for prefix in ["Y", "D"])}
        
        # Structural tokens
        structural = {"[", "]", "{", "}", "=", ",", "(", ")", "#"}
        
        # Special tokens
        special = {"<START>", "<END>", "<PAD>", "<UNK>"}
        
        # Boolean values
        boolean = {"true", "false"}
        
        # Other enum values
        other_enums = {"OPEN_END"}
        
        return {
            "actions": actions,
            "params": params,
            "bay_kinds": bay_kinds,
            "breaker_kinds": breaker_kinds,
            "disconnector_kinds": disconnector_kinds,
            "transformer_kinds": transformer_kinds,
            "line_kinds": line_kinds,
            "bus_ids": bus_ids,
            "breaker_ids": breaker_ids,
            "coupler_ids": coupler_ids,
            "bay_ids": bay_ids,
            "disconnector_ids": disconnector_ids,
            "transformer_ids": transformer_ids,
            "line_ids": line_ids,
            "voltages": voltages,
            "mva_values": mva_values,
            "percentz_values": percentz_values,
            "thermal_values": thermal_values,
            "ka_values": ka_values,
            "counts": counts,
            "reliability_targets": reliability_targets,
            "vector_groups": vector_groups,
            "structural": structural,
            "special": special,
            "boolean": boolean,
            "other_enums": other_enums
        }
    
    def print_stats(self):
        """Print vocabulary statistics"""
        token_sets = self.get_token_sets()
        
        print(f"Total vocabulary size: {self.size()}")
        print("\nToken distribution:")
        for category, tokens in token_sets.items():
            count = len(tokens & set(self.vocab))  # Only count tokens actually in vocab
            print(f"  {category}: {count}")
        
        print(f"\nFirst 10 tokens: {self.vocab[:10]}")
        print(f"Last 10 tokens: {self.vocab[-10:]}")

if __name__ == "__main__":
    # Test the vocabulary loader
    vocab = ActionVocabulary()
    vocab.print_stats()
    
    # Test encoding/decoding
    test_tokens = ["<START>", "ADD_BUS", "id=", "bus100", "<END>"]
    encoded = vocab.encode(test_tokens)
    decoded = vocab.decode(encoded)
    
    print("\nTest encoding/decoding:")
    print(f"Original: {test_tokens}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
