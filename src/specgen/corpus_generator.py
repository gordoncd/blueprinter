"""
corpus_generator.py

Convert realistic substation specifications into action sequence training corpus.
This bridges the gap between engineering specs and trainable token sequences.

Author: Gordon Doore
Date Created: 2025-08-19
"""

import json
import random
from typing import Dict, List
from pathlib import Path

class ActionSequenceGenerator:
    """Convert substation specifications to action sequences"""
    
    def __init__(self):
        # Map spec values to vocabulary tokens
        self.voltage_mapping = {
            500: "345KV",    # Map to closest vocab
            345: "345KV",
            230: "220KV",    # Map to closest vocab  
            220: "220KV",
            138: "138KV",
            115: "115KV",
            69: "69KV",
            66: "66KV",
            46: "46KV",
            34.5: "34.5KV",
            25: "25KV",
            13.8: "13.8KV",
            4.16: "13.8KV"   # Map to closest vocab
        }
        
        self.mva_mapping = {
            1000: "150MVA",  # Map to closest vocab
            750: "150MVA",
            500: "150MVA",
            300: "150MVA",
            200: "150MVA",
            150: "150MVA",
            100: "100MVA",
            75: "75MVA",
            50: "50MVA",
            37.5: "50MVA",   # Map to closest vocab
            25: "50MVA",
            15: "50MVA",
            10: "50MVA"
        }
        
        self.protection_mapping = {
            "basic": "basic",
            "standard": "standard", 
            "high": "enhanced",
            "critical": "enhanced"
        }
        
        self.scheme_mapping = {
            "single_bus": "SINGLE_BUSBAR",
            "double_bus": "DOUBLE_BUSBAR",
            "ring": "DOUBLE_BUSBAR",      # Map to closest vocab
            "breaker_half": "DOUBLE_BUSBAR",
            "main_transfer": "DOUBLE_BUSBAR",
            "radial": "SINGLE_BUSBAR"
        }
    
    def get_vocab_voltage(self, voltage: float) -> str:
        """Map any voltage to closest vocabulary token"""
        return self.voltage_mapping.get(voltage, "220KV")  # Default fallback
    
    def get_vocab_mva(self, mva: float) -> str:  
        """Map any MVA to closest vocabulary token"""
        return self.mva_mapping.get(mva, "100MVA")  # Default fallback
    
    def spec_to_action_sequence(self, spec_data: Dict) -> List[str]:
        """Convert a single specification to action sequence"""
        
        intent = spec_data["intent"]
        spec = spec_data["spec"]
        
        tokens = ["<START>"]
        
        # Get mapped vocabulary values
        hv_voltage = self.get_vocab_voltage(intent["from_kv"])
        lv_voltage = self.get_vocab_voltage(intent["to_kv"])
        transformer_mva = self.get_vocab_mva(intent["rating_mva"])
        protection_level = self.protection_mapping.get(intent.get("protection", "standard"), "standard")
        hv_scheme = intent.get("scheme_hv", "single_bus")
        
        # Add buses based on scheme
        if hv_scheme in ["double_bus", "double_busbar", "ring", "breaker_half"]:
            tokens.extend([
                "ADD_BUS", hv_voltage, "HV", "MAIN", "BUS_101",
                "ADD_BUS", hv_voltage, "HV", "MAIN", "BUS_102"
            ])
            main_hv_bus = "BUS_101"
        else:
            tokens.extend([
                "ADD_BUS", hv_voltage, "HV", "MAIN", "BUS_101"
            ])
            main_hv_bus = "BUS_101"
        
        # Add LV bus
        tokens.extend([
            "ADD_BUS", lv_voltage, "LV", "MAIN", "BUS_103"
        ])
        
        # Add transformer
        tokens.extend([
            "ADD_TRANSFORMER", "TWO_WINDING", hv_voltage,
            lv_voltage, transformer_mva, "TX_201"
        ])
        
        # Add transformer bay with protection based on reliability
        tokens.extend([
            "ADD_BAY", "HV", hv_voltage, "TRANSFORMER_BAY", "BAY_301"
        ])
        
        # Add protection equipment based on protection level
        if protection_level == "enhanced":
            tokens.extend([
                "APPEND_STEP", "BAY_301", "BUS_ISOLATOR",
                "APPEND_STEP", "BAY_301", "CT",  # Enhanced protection
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
        
        # Add feeder bays
        num_feeders = intent.get("lv_feeders", 2)
        feeder_types = spec.get("feeder_types", ["residential"] * num_feeders)
        
        for i in range(min(num_feeders, 8)):  # Cap at 8 feeders for vocab limits
            bay_id = f"BAY_{302 + i}"
            
            tokens.extend([
                "ADD_BAY", "LV", lv_voltage, "FEEDER_BAY", bay_id
            ])
            
            # Vary protection based on feeder type and protection level
            feeder_type = feeder_types[i] if i < len(feeder_types) else "residential"
            
            if feeder_type == "industrial" or protection_level == "enhanced":
                # Industrial feeders or enhanced protection get more equipment
                tokens.extend([
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR",
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "CT"
                ])
            elif feeder_type == "commercial" or protection_level == "standard":
                # Commercial feeders get standard protection
                tokens.extend([
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR", 
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "LINE_ISOLATOR"
                ])
            else:
                # Residential feeders get basic protection
                tokens.extend([
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR"
                ])
        
        # Add connections
        tokens.extend([
            "CONNECT_BUS", main_hv_bus, "BAY_301",
            "CONNECT_TX_HV", "TX_201", main_hv_bus,
            "CONNECT_TX_LV", "TX_201", "BUS_103"
        ])
        
        # Add spec emission
        tokens.append("EMIT_SPEC")
        tokens.append("<END>")
        
        return tokens
    
    def generate_corpus_from_data(self, data_file: str, output_file: str = None) -> List[Dict]:
        """Convert realistic substation data to action sequence corpus"""
        
        print(f"Loading realistic data from {data_file}...")
        
        with open(data_file, 'r') as f:
            realistic_data = json.load(f)
        
        print(f"Converting {len(realistic_data)} specifications to action sequences...")
        
        corpus = []
        
        for i, spec_data in enumerate(realistic_data):
            if i % 100 == 0:
                print(f"Converted {i}/{len(realistic_data)}")
            
            try:
                # Generate action sequence
                action_sequence = self.spec_to_action_sequence(spec_data)
                
                # Create training example
                corpus_entry = {
                    "id": spec_data["id"],
                    "intent": spec_data["intent"],
                    "spec": spec_data["spec"],
                    "action_sequence": action_sequence,
                    "sequence_length": len(action_sequence)
                }
                
                corpus.append(corpus_entry)
                
            except Exception as e:
                print(f"Error processing {spec_data['id']}: {e}")
                continue
        
        print(f"Successfully converted {len(corpus)} examples")
        
        # Save corpus if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(corpus, f, indent=2)
            print(f"Saved corpus to {output_file}")
        
        return corpus
    
    def analyze_corpus(self, corpus: List[Dict]):
        """Analyze the generated corpus"""
        print(f"\n=== Corpus Analysis ({len(corpus)} examples) ===")
        
        # Sequence length statistics
        lengths = [entry["sequence_length"] for entry in corpus]
        print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
        
        # Token frequency analysis
        all_tokens = []
        for entry in corpus:
            all_tokens.extend(entry["action_sequence"])
        
        from collections import Counter
        token_counts = Counter(all_tokens)
        
        print(f"Total tokens: {len(all_tokens)}")
        print(f"Unique tokens: {len(token_counts)}")
        print("Most frequent tokens:")
        for token, count in token_counts.most_common(10):
            print(f"  {token}: {count}")
        
        # Check for unknown tokens
        unknown_tokens = [token for token in token_counts if "<UNK>" in token]
        if unknown_tokens:
            print(f"Warning: Found {len(unknown_tokens)} unknown tokens")
        
        # Substation type distribution
        types = [entry["spec"]["substation_type"] for entry in corpus]
        type_counts = Counter(types)
        print("Substation types:", dict(type_counts))
    
    def sample_corpus(self, corpus: List[Dict], num_samples: int = 5):
        """Show sample entries from corpus"""
        print("\n=== Sample Corpus Entries ===")
        
        samples = random.sample(corpus, min(num_samples, len(corpus)))
        
        for i, entry in enumerate(samples):
            print(f"\nSample {i+1}: {entry['id']}")
            print(f"  Type: {entry['spec']['substation_type']}")
            print(f"  Voltage: {entry['intent']['from_kv']}kV -> {entry['intent']['to_kv']}kV")
            print(f"  Power: {entry['intent']['rating_mva']} MVA")
            print(f"  Sequence ({entry['sequence_length']} tokens):")
            
            # Show sequence in readable format
            tokens = entry["action_sequence"]
            sequence_str = " ".join(tokens[:15])  # First 15 tokens
            if len(tokens) > 15:
                sequence_str += " ... " + " ".join(tokens[-5:])  # Last 5 tokens
            print(f"    {sequence_str}")


def main():
    """Generate training corpus from realistic substation data"""
    
    generator = ActionSequenceGenerator()
    
    # Convert realistic data to action sequences
    data_file = "realistic_substation_data.json"
    corpus_file = "substation_action_corpus.json"
    
    if not Path(data_file).exists():
        print(f"Error: {data_file} not found. Run data_generator.py first.")
        return
    
    # Generate corpus
    corpus = generator.generate_corpus_from_data(data_file, corpus_file)
    
    # Analyze results
    generator.analyze_corpus(corpus)
    generator.sample_corpus(corpus)
    
    print(f"\n✅ Training corpus ready: {corpus_file}")
    print(f"   {len(corpus)} action sequences generated")
    print("   Ready for model training!")


if __name__ == "__main__":
    main()
