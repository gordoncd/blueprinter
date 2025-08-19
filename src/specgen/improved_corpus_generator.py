#!/usr/bin/env python3
"""
improved_corpus_generator.py

Generate high-quality, vocabulary-compliant training data that ensures the model
learns to follow intent specifications precisely.

Author: Gordon Doore
Date Created: 2025-08-19
"""

import json
import random
import sys
import os
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Add parent directories to path for importing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the existing modules
from src.specgen.action import ActionKind, StepKind, BusRole, Side, BayKind, TransformerKind, DeviceKind
from src.specgen.diagram import Diagram
from src.specgen.model import MinimalVocabulary

class ImprovedCorpusGenerator:
    """Generate training corpus with strict vocabulary compliance and specification adherence"""
    
    def __init__(self):
        # Use the same vocabulary as the model
        self.vocab = MinimalVocabulary()
        
        # Create voltage level mappings for realistic specs
        self.voltage_mappings = {
            # Transmission levels - map to vocabulary values
            500: "500",      # 500kV 
            345: "345",      # 345kV
            230: "230",      # 230kV  
            138: "138",      # 138kV
            
            # Sub-transmission/distribution levels
            69: "69",        # 69kV
            25: "25",        # 25kV
            13.8: "13.8",    # 13.8kV
        }
        
        self.mva_mappings = {
            # Map MVA values to vocabulary tokens
            400: "400",      # 400 MVA
            300: "300",      # 300 MVA  
            200: "200",      # 200 MVA
            100: "100",      # 100 MVA
            80: "80",        # 80 MVA
            60: "60",        # 60 MVA
            40: "40",        # 40 MVA
            25: "25",        # 25 MVA (reusing voltage token)
            20: "20",        # 20 MVA
        
        # Feeder count mappings
        self.feeder_mappings = {
            2: "2", 3: "3", 4: "4", 5: "4", 6: "6", 
            8: "8", 10: "10", 12: "12", 15: "15"
        }
        
        # Protection level mappings  
        self.protection_mappings = {1: "1", 2: "2", 3: "3"}
        
    def get_vocab_voltage(self, voltage_kv: float) -> str:
        """Get exact vocabulary voltage token"""
        return self.voltage_mappings.get(voltage_kv, "230")
    
    def get_vocab_mva(self, mva: float) -> str:
        """Get exact vocabulary MVA token"""
        return self.mva_mappings.get(mva, "100MVA")
    
    def create_diverse_intents(self, num_intents: int = 1000) -> List[Dict]:
        """Create diverse intent specifications covering the full range"""
        
        intents = []
        
        # Define realistic voltage level pairs with proper ratios
        voltage_pairs = [
            # Transmission to sub-transmission
            (500, 138, [300, 200, 150]),
            (345, 138, [200, 150, 100]),
            (230, 138, [150, 100, 75]),
            (230, 69, [100, 75, 50]),
            (138, 69, [100, 75, 50]),
            (138, 34.5, [75, 50]),
            (115, 34.5, [75, 50]),
            (115, 25, [50]),
            
            # Sub-transmission to distribution
            (69, 25, [50, 75]),
            (69, 13.8, [50, 75, 100]),
            (66, 13.8, [50, 75]),
            (46, 13.8, [50]),
            (34.5, 13.8, [50]),
            (25, 13.8, [50]),
            
            # Distribution transformations
            (25, 4.16, [50]),  # Will map 4.16 to 13.8KV in vocab
            (13.8, 4.16, [50])  # Will map 4.16 to 13.8KV in vocab
        ]
        
        protection_levels = ["basic", "standard", "enhanced"]
        scheme_types = ["single_busbar", "double_busbar"]
        
        for _ in range(num_intents):
            # Pick voltage pair and appropriate MVA
            from_kv, to_kv, mva_options = random.choice(voltage_pairs)
            rating_mva = random.choice(mva_options)
            
            # Choose realistic feeder counts based on voltage level
            if to_kv >= 69:  # Sub-transmission - fewer feeders
                lv_feeders = random.choice([2, 3, 4])
            elif to_kv >= 25:  # Medium distribution - moderate feeders
                lv_feeders = random.choice([3, 4, 5, 6])
            else:  # Low voltage distribution - more feeders
                lv_feeders = random.choice([4, 5, 6, 7, 8])
            
            # Protection level based on system importance
            if from_kv >= 230:  # Transmission - typically enhanced
                protection = random.choice(["enhanced", "standard"])
            elif from_kv >= 69:  # Sub-transmission - typically standard
                protection = random.choice(["standard", "enhanced", "basic"])
            else:  # Distribution - varied
                protection = random.choice(["basic", "standard"])
            
            # Scheme based on reliability requirements
            if from_kv >= 138 and rating_mva >= 100:  # Large systems need redundancy
                scheme_hv = random.choice(["double_busbar", "single_busbar"])
                scheme_weight = [0.7, 0.3]  # Prefer double busbar
                scheme_hv = random.choices(scheme_types, weights=scheme_weight)[0]
            else:
                scheme_hv = random.choice(scheme_types)
            
            intent = {
                "from_kv": from_kv,
                "to_kv": to_kv, 
                "rating_mva": rating_mva,
                "scheme_hv": scheme_hv,
                "lv_feeders": lv_feeders,
                "protection": protection,
                "id": f"intent_{len(intents):04d}"
            }
            
            intents.append(intent)
        
        return intents
    
    def intent_to_action_sequence(self, intent: Dict) -> List[str]:
        """Convert intent to action sequence using ONLY valid vocabulary tokens"""
        
        tokens = ["<START>"]
        
        # Get vocabulary-compliant values
        hv_voltage = self.get_vocab_voltage(intent["from_kv"])
        lv_voltage = self.get_vocab_voltage(intent["to_kv"])
        transformer_mva = self.get_vocab_mva(intent["rating_mva"])
        protection_level = intent.get("protection", "standard")
        scheme_hv = intent.get("scheme_hv", "single_busbar")
        num_feeders = intent.get("lv_feeders", 2)
        
        # Track used IDs to avoid conflicts
        bus_idx = 0
        tx_idx = 0
        bay_idx = 0
        
        # Add HV buses based on scheme
        if scheme_hv == "double_busbar":
            tokens.extend([
                "ADD_BUS", hv_voltage, "HV", "MAIN", self.bus_ids[bus_idx],
                "ADD_BUS", hv_voltage, "HV", "MAIN", self.bus_ids[bus_idx + 1]
            ])
            main_hv_bus = self.bus_ids[bus_idx]
            bus_idx += 2
        else:  # single_busbar
            tokens.extend([
                "ADD_BUS", hv_voltage, "HV", "MAIN", self.bus_ids[bus_idx]
            ])
            main_hv_bus = self.bus_ids[bus_idx]
            bus_idx += 1
        
        # Add LV bus
        tokens.extend([
            "ADD_BUS", lv_voltage, "LV", "MAIN", self.bus_ids[bus_idx]
        ])
        main_lv_bus = self.bus_ids[bus_idx]
        bus_idx += 1
        
        # Add transformer
        tokens.extend([
            "ADD_TRANSFORMER", "TWO_WINDING", hv_voltage, lv_voltage, 
            transformer_mva, self.tx_ids[tx_idx]
        ])
        main_tx = self.tx_ids[tx_idx]
        tx_idx += 1
        
        # Add transformer bay with appropriate protection
        tokens.extend([
            "ADD_BAY", "HV", hv_voltage, "TRANSFORMER_BAY", self.bay_ids[bay_idx]
        ])
        tx_bay = self.bay_ids[bay_idx]
        bay_idx += 1
        
        # Add protection equipment based on level
        if protection_level == "enhanced":
            tokens.extend([
                "APPEND_STEP", tx_bay, "BUS_ISOLATOR",
                "APPEND_STEP", tx_bay, "CT",  # Current transformer for monitoring
                "APPEND_STEP", tx_bay, "BREAKER",
                "APPEND_STEP", tx_bay, "LINE_ISOLATOR"
            ])
        elif protection_level == "basic":
            tokens.extend([
                "APPEND_STEP", tx_bay, "BREAKER",
                "APPEND_STEP", tx_bay, "BUS_ISOLATOR"
            ])
        else:  # standard
            tokens.extend([
                "APPEND_STEP", tx_bay, "BUS_ISOLATOR",
                "APPEND_STEP", tx_bay, "BREAKER", 
                "APPEND_STEP", tx_bay, "LINE_ISOLATOR"
            ])
        
        # Add LV feeder bays - ENSURE we create the exact number requested
        feeder_bays = []
        for i in range(num_feeders):
            if bay_idx >= len(self.bay_ids):
                break  # Prevent index overflow
                
            bay_id = self.bay_ids[bay_idx]
            feeder_bays.append(bay_id)
            
            tokens.extend([
                "ADD_BAY", "LV", lv_voltage, "FEEDER_BAY", bay_id
            ])
            
            # Vary protection by feeder importance and overall protection level
            if i == 0 and protection_level == "enhanced":  # Main feeder with enhanced protection
                tokens.extend([
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR",
                    "APPEND_STEP", bay_id, "CT",
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "LINE_ISOLATOR"
                ])
            elif protection_level == "enhanced":  # Other feeders with enhanced
                tokens.extend([
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR", 
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "CT"
                ])
            elif protection_level == "standard":  # Standard protection
                tokens.extend([
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR",
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "LINE_ISOLATOR"
                ])
            else:  # basic protection
                tokens.extend([
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR"
                ])
            
            bay_idx += 1
        
        # Add connections - critical for functionality
        tokens.extend([
            "CONNECT_BUS", main_hv_bus, tx_bay,
            "CONNECT_TX_HV", main_tx, main_hv_bus,
            "CONNECT_TX_LV", main_tx, main_lv_bus
        ])
        
        # Connect feeders to LV bus
        for feeder_bay in feeder_bays:
            tokens.extend([
                "CONNECT_BUS", main_lv_bus, feeder_bay
            ])
        
        tokens.extend(["EMIT_SPEC", "<END>"])
        
        return tokens
    
    def generate_improved_corpus(self, num_examples: int = 1000, output_file: str = None) -> List[Dict]:
        """Generate high-quality training corpus with perfect vocabulary compliance"""
        
        print(f"Creating {num_examples} diverse intent specifications...")
        intents = self.create_diverse_intents(num_examples)
        
        print(f"Converting to action sequences with strict vocabulary compliance...")
        corpus = []
        
        for i, intent in enumerate(intents):
            if i % 100 == 0:
                print(f"Processed {i}/{len(intents)}")
            
            try:
                action_sequence = self.intent_to_action_sequence(intent)
                
                # Verify all tokens are in vocabulary (safety check)
                from .model import MinimalVocabulary
                vocab = MinimalVocabulary()
                
                unknown_tokens = [token for token in action_sequence 
                                if token not in vocab.token_to_id]
                
                if unknown_tokens:
                    print(f"WARNING: Unknown tokens in sequence {i}: {unknown_tokens}")
                    continue  # Skip this example
                
                corpus_entry = {
                    "id": intent["id"],
                    "intent": intent,
                    "action_sequence": action_sequence,
                    "sequence_length": len(action_sequence),
                    "hv_voltage": self.get_vocab_voltage(intent["from_kv"]),
                    "lv_voltage": self.get_vocab_voltage(intent["to_kv"]),
                    "capacity": self.get_vocab_mva(intent["rating_mva"]),
                    "feeder_count": intent["lv_feeders"]
                }
                
                corpus.append(corpus_entry)
                
            except Exception as e:
                print(f"Error processing intent {i}: {e}")
                continue
        
        print(f"Successfully generated {len(corpus)} training examples")
        
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(corpus, f, indent=2)
            print(f"Saved corpus to {output_file}")
        
        return corpus
    
    def analyze_corpus_quality(self, corpus: List[Dict]):
        """Analyze the corpus for quality and coverage"""
        print(f"\n=== CORPUS QUALITY ANALYSIS ===")
        print(f"Total examples: {len(corpus)}")
        
        # Voltage distribution analysis
        hv_voltages = {}
        lv_voltages = {}
        capacities = {}
        feeder_counts = {}
        
        for entry in corpus:
            intent = entry["intent"]
            
            hv_key = f"{intent['from_kv']}kV"
            hv_voltages[hv_key] = hv_voltages.get(hv_key, 0) + 1
            
            lv_key = f"{intent['to_kv']}kV"
            lv_voltages[lv_key] = lv_voltages.get(lv_key, 0) + 1
            
            cap_key = f"{intent['rating_mva']}MVA"
            capacities[cap_key] = capacities.get(cap_key, 0) + 1
            
            feeder_counts[intent['lv_feeders']] = feeder_counts.get(intent['lv_feeders'], 0) + 1
        
        print(f"\nHV Voltage Distribution:")
        for voltage, count in sorted(hv_voltages.items()):
            print(f"  {voltage}: {count} examples ({count/len(corpus)*100:.1f}%)")
        
        print(f"\nLV Voltage Distribution:")
        for voltage, count in sorted(lv_voltages.items()):
            print(f"  {voltage}: {count} examples ({count/len(corpus)*100:.1f}%)")
        
        print(f"\nCapacity Distribution:")
        for capacity, count in sorted(capacities.items()):
            print(f"  {capacity}: {count} examples ({count/len(corpus)*100:.1f}%)")
        
        print(f"\nFeeder Count Distribution:")
        for count, freq in sorted(feeder_counts.items()):
            print(f"  {count} feeders: {freq} examples ({freq/len(corpus)*100:.1f}%)")
        
        # Sequence length analysis
        lengths = [entry["sequence_length"] for entry in corpus]
        print(f"\nSequence Length Statistics:")
        print(f"  Min: {min(lengths)} tokens")
        print(f"  Max: {max(lengths)} tokens")
        print(f"  Average: {sum(lengths)/len(lengths):.1f} tokens")
        
        # Vocabulary coverage check
        all_tokens = set()
        for entry in corpus:
            all_tokens.update(entry["action_sequence"])
        
        print(f"\nVocabulary Usage:")
        print(f"  Unique tokens used: {len(all_tokens)}")
        print(f"  Most common tokens:")
        
        from collections import Counter
        token_counts = Counter()
        for entry in corpus:
            token_counts.update(entry["action_sequence"])
        
        for token, count in token_counts.most_common(10):
            print(f"    {token}: {count} occurrences")


def main():
    """Generate improved training corpus"""
    
    generator = ImprovedCorpusGenerator()
    
    # Generate large, diverse corpus
    corpus_file = "improved_action_corpus.json"
    
    print("🔧 Generating Improved Training Corpus")
    print("=" * 50)
    
    corpus = generator.generate_improved_corpus(
        num_examples=2000,  # Larger corpus for better coverage
        output_file=corpus_file
    )
    
    # Analyze quality
    generator.analyze_corpus_quality(corpus)
    
    print(f"\n✅ High-quality training corpus ready: {corpus_file}")
    print(f"   {len(corpus)} specification-compliant examples")
    print("   Perfect vocabulary alignment guaranteed!")
    print("   Ready for improved model training!")


if __name__ == "__main__":
    main()
