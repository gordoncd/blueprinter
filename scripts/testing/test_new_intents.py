#!/usr/bin/env python3
"""
Test the trained model with new intent vectors to show grammar enforcement and intent alignment
"""

import sys
import os
import json
from typing import Dict

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from src.specgen.model import MinimalActionModel, MinimalConfig

def test_model_on_new_intents():
    """Test the trained model on completely new intent vectors"""
    
    print("🎯 TESTING TRAINED MODEL ON NEW INTENT VECTORS")
    print("=" * 70)
    
    # Load trained model
    config = MinimalConfig()
    model = MinimalActionModel(config)
    
    try:
        checkpoint = torch.load('trained_model.pt', map_location='cpu')
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model.eval()
        print("✅ Trained model loaded successfully")
    except:
        print("❌ Could not load trained model")
        return
    
    # New test intents not seen during training
    new_test_intents = [
        {
            "from_kv": 138,
            "to_kv": 34.5,
            "rating_mva": 75,
            "scheme_hv": "double_busbar",
            "lv_feeders": 3,
            "protection": "enhanced",
            "description": "138kV to 34.5kV transmission substation"
        },
        {
            "from_kv": 115,
            "to_kv": 13.8,
            "rating_mva": 100,
            "scheme_hv": "single_bus",
            "lv_feeders": 6,
            "protection": "standard",
            "description": "115kV industrial substation"
        },
        {
            "from_kv": 220,
            "to_kv": 25,
            "rating_mva": 50,
            "scheme_hv": "single_bus",
            "lv_feeders": 2,
            "protection": "basic",
            "description": "220kV to 25kV distribution substation"
        }
    ]
    
    for i, intent in enumerate(new_test_intents, 1):
        print(f"\n{'='*50}")
        print(f"🔸 NEW TEST {i}: {intent['description'].upper()}")
        print(f"{'='*50}")
        
        print(f"\n📋 Intent Specification:")
        print(f"  • From: {intent['from_kv']}kV")
        print(f"  • To: {intent['to_kv']}kV")
        print(f"  • Capacity: {intent['rating_mva']}MVA")
        print(f"  • HV Scheme: {intent['scheme_hv']}")
        print(f"  • LV Feeders: {intent['lv_feeders']}")
        print(f"  • Protection: {intent['protection']}")
        
        # Generate with different design philosophies
        philosophies = ["conservative", "standard", "economical"]
        
        for philosophy in philosophies:
            print(f"\n🎨 {philosophy.upper()} DESIGN APPROACH:")
            print("-" * 30)
            
            try:
                # Generate sequence
                sequence = model.generate_from_intent(
                    intent=intent,
                    design_philosophy=philosophy,
                    temperature=0.7,
                    max_length=200
                )
                
                if sequence:
                    print(f"Generated {len(sequence)} tokens")
                    
                    # Format and display the sequence
                    formatted_sequence = format_action_sequence_compact(sequence)
                    
                    print("\n🔧 Action Sequence:")
                    for action in formatted_sequence:
                        print(f"  {action}")
                    
                    # Check grammar compliance
                    full_sequence = ["<START>"] + sequence + ["<END>"]
                    is_complete = model.grammar.is_sequence_complete(full_sequence)
                    
                    print(f"\n📊 Grammar Check: {'✅ Valid' if is_complete else '❌ Invalid'}")
                    
                    # Count key components
                    bus_count = sequence.count("ADD_BUS")
                    tx_count = sequence.count("ADD_TRANSFORMER")
                    bay_count = sequence.count("ADD_BAY")
                    feeder_count = sequence.count("FEEDER_BAY")
                    
                    print(f"📈 Configuration: {bus_count} buses, {tx_count} transformers, {bay_count} bays ({feeder_count} feeders)")
                    
                else:
                    print("❌ Failed to generate sequence")
                    
            except Exception as e:
                print(f"❌ Error: {str(e)}")
    
    # Test grammar enforcement specifically
    print(f"\n\n{'='*70}")
    print("🛡️ GRAMMAR ENFORCEMENT DEMONSTRATION")
    print(f"{'='*70}")
    
    test_intent = {
        "from_kv": 69,
        "to_kv": 13.8,
        "rating_mva": 50,
        "scheme_hv": "single_bus",
        "lv_feeders": 2,
        "protection": "standard"
    }
    
    print(f"\n🎯 Testing grammar enforcement with intent: {test_intent['from_kv']}kV → {test_intent['to_kv']}kV")
    
    # Test different temperatures to show grammar is maintained
    temperatures = [0.1, 0.8, 1.5]
    
    for temp in temperatures:
        print(f"\n🌡️ Temperature {temp} (randomness level):")
        print("-" * 25)
        
        try:
            candidates = model.generate(
                intent=test_intent,
                max_length=150,
                temperature=temp,
                num_candidates=2
            )
            
            for j, sequence in enumerate(candidates, 1):
                print(f"\n  Candidate {j} ({len(sequence)} tokens):")
                
                # Quick format for display
                preview = []
                for k in range(0, min(len(sequence), 15), 5):
                    chunk = " ".join(sequence[k:k+5])
                    preview.append(f"    {chunk}")
                
                for line in preview:
                    print(line)
                
                if len(sequence) > 15:
                    print(f"    ... and {len(sequence) - 15} more tokens")
                
                # Verify grammar
                full_seq = ["<START>"] + sequence + ["<END>"]
                valid = model.grammar.is_sequence_complete(full_seq)
                print(f"    Grammar: {'✅ Valid' if valid else '❌ Invalid'}")
                
        except Exception as e:
            print(f"    ❌ Error at temp {temp}: {str(e)}")

def format_action_sequence_compact(sequence):
    """Format action sequence in a more compact format"""
    formatted = []
    i = 0
    
    while i < len(sequence):
        token = sequence[i]
        
        if token == "ADD_BUS" and i + 4 < len(sequence):
            formatted.append(f"ADD_BUS {sequence[i+1]} {sequence[i+2]} {sequence[i+3]} {sequence[i+4]}")
            i += 5
        elif token == "ADD_TRANSFORMER" and i + 5 < len(sequence):
            formatted.append(f"ADD_TRANSFORMER {sequence[i+1]} {sequence[i+2]} → {sequence[i+3]} {sequence[i+4]} {sequence[i+5]}")
            i += 6
        elif token == "ADD_BAY" and i + 4 < len(sequence):
            formatted.append(f"ADD_BAY {sequence[i+1]} {sequence[i+2]} {sequence[i+3]} {sequence[i+4]}")
            i += 5
        elif token == "APPEND_STEP" and i + 2 < len(sequence):
            formatted.append(f"  └─ APPEND_STEP {sequence[i+1]} {sequence[i+2]}")
            i += 3
        elif token in ["CONNECT_BUS", "CONNECT_TX_HV", "CONNECT_TX_LV"] and i + 2 < len(sequence):
            formatted.append(f"{token} {sequence[i+1]} ↔ {sequence[i+2]}")
            i += 3
        else:
            formatted.append(token)
            i += 1
    
    return formatted

if __name__ == "__main__":
    test_model_on_new_intents()
