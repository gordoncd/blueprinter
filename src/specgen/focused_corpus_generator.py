"""
focused_corpus_generator.py

Creates a focused, balanced training corpus with explicit voltage pairings
to help the model learn correct input/output voltage alignment.

Author: Gordon Doore
Date Created: 2025-08-19
"""

import json
import random
import sys
import os
from typing import Dict, List

# Add path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.specgen.model import MinimalVocabulary

class FocusedCorpusGenerator:
    """Generate focused training corpus with balanced voltage pairings"""
    
    def __init__(self):
        self.vocab = MinimalVocabulary()
        
        # Define realistic voltage pairings with balanced representation
        self.voltage_pairings = [
            # Common distribution pairings
            (138.0, 13.8, [75, 100, 150]),  # Most common transmission to distribution
            (115.0, 13.8, [50, 75, 100]),   # Common distribution
            (69.0, 13.8, [50, 75]),         # Distribution substation
            (138.0, 34.5, [75, 100, 150]),  # Transmission to sub-transmission
            (138.0, 69.0, [100, 150]),      # Transmission stepping down
            (220.0, 138.0, [150, 300]),     # High voltage transmission
            (345.0, 138.0, [300]),          # Major transmission
            (115.0, 34.5, [50, 75, 100]),   # Medium voltage
            (34.5, 13.8, [10, 25, 50, 75]), # Small distribution - expanded MVA
            (25.0, 13.8, [25, 50, 75]),     # Industrial distribution - expanded MVA
            (138.0, 25.0, [75, 100]),       # Industrial transmission
            (115.0, 25.0, [50, 75]),        # Industrial stepping down
            # New missing voltage pairings from validator
            (25.0, 4.16, [10, 25]),         # Industrial plant substation
            (230.0, 138.0, [300]),          # Major transmission (230kV version)
        ]
    
    def create_balanced_intent(self, from_kv: float, to_kv: float, mva_options: List[int]) -> Dict:
        """Create intent with balanced parameters"""
        
        # Choose MVA rating
        rating_mva = random.choice(mva_options)
        
        # Determine feeder count based on size
        if rating_mva >= 150:
            lv_feeders = random.choice([2, 3, 4])  # Large substations have fewer feeders
        elif rating_mva >= 75:
            lv_feeders = random.choice([3, 4, 6])  # Medium substations
        else:
            lv_feeders = random.choice([4, 6, 8])  # Small substations have more feeders
        
        # Choose configuration based on criticality
        if rating_mva >= 150:
            scheme_hv = random.choice(["double_busbar", "double_busbar", "single_busbar"])  # Bias toward redundancy
            protection = random.choice(["enhanced", "standard"])
        else:
            scheme_hv = random.choice(["single_busbar", "double_busbar"])
            protection = random.choice(["standard", "basic"])
        
        # Ensure voltage tokens match vocabulary exactly
        from_kv_token = f"{from_kv}KV" if from_kv != int(from_kv) else f"{int(from_kv)}KV"
        to_kv_token = f"{to_kv}KV" if to_kv != int(to_kv) else f"{int(to_kv)}KV"
        mva_token = f"{rating_mva}MVA"
        
        # Validate tokens are in vocabulary
        if from_kv_token not in self.vocab.vocab:
            print(f"Warning: {from_kv_token} not in vocabulary")
        if to_kv_token not in self.vocab.vocab:
            print(f"Warning: {to_kv_token} not in vocabulary")
        if mva_token not in self.vocab.vocab:
            print(f"Warning: {mva_token} not in vocabulary")
        
        return {
            "from_kv": from_kv,
            "to_kv": to_kv,
            "rating_mva": rating_mva,
            "scheme_hv": scheme_hv,
            "lv_feeders": lv_feeders,
            "protection": protection,
            "_vocab_hv": from_kv_token,
            "_vocab_lv": to_kv_token,
            "_vocab_mva": mva_token
        }
    
    def generate_action_sequence(self, intent: Dict) -> List[str]:
        """Generate canonical action sequence for the intent"""
        
        tokens = ["<START>"]
        
        # Use vocab tokens from intent
        from_kv = intent["_vocab_hv"]
        to_kv = intent["_vocab_lv"]
        mva = intent["_vocab_mva"]
        
        # Add buses based on scheme
        if intent.get("scheme_hv") == "double_busbar":
            tokens.extend([
                "ADD_BUS", from_kv, "HV", "MAIN", "BUS_101",
                "ADD_BUS", from_kv, "HV", "MAIN", "BUS_102"
            ])
            main_bus = "BUS_101"
        else:
            tokens.extend([
                "ADD_BUS", from_kv, "HV", "MAIN", "BUS_101"
            ])
            main_bus = "BUS_101"
        
        # Add LV bus
        tokens.extend([
            "ADD_BUS", to_kv, "LV", "MAIN", "BUS_103"
        ])
        
        # Add transformer
        tokens.extend([
            "ADD_TRANSFORMER", "TWO_WINDING", from_kv, to_kv, mva, "TX_201"
        ])
        
        # Add transformer bay
        tokens.extend([
            "ADD_BAY", "HV", from_kv, "TRANSFORMER_BAY", "BAY_301"
        ])
        
        # Protection sequence based on protection level
        protection_level = intent.get("protection", "standard")
        if protection_level == "enhanced":
            tokens.extend([
                "APPEND_STEP", "BAY_301", "BUS_ISOLATOR",
                "APPEND_STEP", "BAY_301", "CT",
                "APPEND_STEP", "BAY_301", "BREAKER",
                "APPEND_STEP", "BAY_301", "LINE_ISOLATOR"
            ])
        elif protection_level == "basic":
            tokens.extend([
                "APPEND_STEP", "BAY_301", "BREAKER",
                "APPEND_STEP", "BAY_301", "BUS_ISOLATOR"
            ])
        else:  # standard
            tokens.extend([
                "APPEND_STEP", "BAY_301", "BUS_ISOLATOR",
                "APPEND_STEP", "BAY_301", "BREAKER",
                "APPEND_STEP", "BAY_301", "LINE_ISOLATOR"
            ])
        
        # Add feeders
        num_feeders = intent.get("lv_feeders", 2)
        for i in range(num_feeders):
            bay_id = f"BAY_{302+i}"
            tokens.extend([
                "ADD_BAY", "LV", to_kv, "FEEDER_BAY", bay_id
            ])
            
            # Vary feeder protection
            if i == 0 or protection_level == "enhanced":
                tokens.extend([
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR",
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "CT"
                ])
            else:
                tokens.extend([
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR"
                ])
        
        # Add connections
        tokens.extend([
            "CONNECT_BUS", main_bus, "BAY_301",
            "CONNECT_TX_HV", "TX_201", main_bus,
            "CONNECT_TX_LV", "TX_201", "BUS_103"
        ])
        
        tokens.append("<END>")
        
        return tokens
    
    def generate_focused_corpus(self, examples_per_pairing: int = 40) -> List[Dict]:
        """Generate focused corpus with balanced voltage pairings"""
        
        corpus = []
        
        print("Generating focused corpus with balanced voltage pairings...")
        
        for i, (from_kv, to_kv, mva_options) in enumerate(self.voltage_pairings):
            print(f"  Generating {examples_per_pairing} examples for {from_kv}kV -> {to_kv}kV")
            
            for j in range(examples_per_pairing):
                # Create intent
                intent = self.create_balanced_intent(from_kv, to_kv, mva_options)
                
                # Generate action sequence
                action_sequence = self.generate_action_sequence(intent)
                
                # Create corpus entry
                entry = {
                    "id": f"focused_{i:02d}_{j:03d}",
                    "intent": intent,
                    "action_sequence": action_sequence
                }
                
                corpus.append(entry)
        
        print(f"Generated {len(corpus)} focused training examples")
        return corpus

def main():
    """Generate and save focused corpus"""
    
    generator = FocusedCorpusGenerator()
    
    # Generate corpus with 40 examples per voltage pairing
    corpus = generator.generate_focused_corpus(examples_per_pairing=40)
    
    # Shuffle for training diversity
    random.shuffle(corpus)
    
    # Save to file
    output_file = "focused_action_corpus.json"
    with open(output_file, 'w') as f:
        json.dump(corpus, f, indent=2)
    
    print(f"Focused corpus saved to {output_file}")
    
    # Print statistics
    voltage_pairs = set()
    for entry in corpus:
        intent = entry["intent"]
        pair = (intent["_vocab_hv"], intent["_vocab_lv"])
        voltage_pairs.add(pair)
    
    print(f"\nCorpus contains {len(voltage_pairs)} unique voltage pairings:")
    for hv, lv in sorted(voltage_pairs):
        count = sum(1 for entry in corpus if entry["intent"]["_vocab_hv"] == hv and entry["intent"]["_vocab_lv"] == lv)
        print(f"  {hv} -> {lv}: {count} examples")

if __name__ == "__main__":
    main()
