#!/usr/bin/env python3

"""
Test generation after overfitting to clean data examples.
"""

import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from src.specgen.model import MinimalActionModel, MinimalConfig
import torch

def test_generation_after_overfitting():
    print("Testing generation after overfitting to clean examples...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create config 
    config = MinimalConfig(
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=100,
        dropout=0.1,
        learning_rate=5e-4
    )
    
    # Create and train model on clean data
    model = MinimalActionModel(config)
    
    print("Training on clean examples...")
    model.train(data_file="data/clean_dev_data.json", num_epochs=300)  # Heavy overfitting
    
    print("\n" + "="*60)
    print("TESTING GENERATION ON TRAINING EXAMPLES")
    print("="*60)
    
    # Load the clean training examples
    with open('data/clean_dev_data.json', 'r') as f:
        clean_data = json.load(f)
    
    # Test generation on each training example
    for i, example in enumerate(clean_data[:3]):  # Test first 3 examples
        intent = example["intent"]
        original_sequence = example["sequence"]
        
        print(f"\n--- Example {i+1} ---")
        print(f"Intent: {intent['kv_in'][0]} -> {intent['kv_out']}")
        print(f"Original: {' '.join(original_sequence[1:-1])}")  # Remove START/END
        
        # Generate with different temperatures
        print("\nGeneration tests:")
        
        # Conservative (low temperature) - should closely match training
        try:
            conservative = model.generate(intent, temperature=0.1, num_candidates=1)
            print(f"  Conservative (T=0.1): {' '.join(conservative[0])}")
            
            # Check if generated sequences are valid
            from src.specgen.grammar_enforcer import GrammarEnforcer
            from src.specgen.vocabulary import ActionVocabulary
            
            vocab = ActionVocabulary()
            enforcer = GrammarEnforcer(vocab)
            
            full_seq = ["<START>"] + conservative[0] + ["<END>"]
            is_valid, error = enforcer.validate_sequence(full_seq)
            print(f"  Conservative validation: {'✅ VALID' if is_valid else f'❌ {error}'}")
            
        except Exception as e:
            print(f"  Conservative failed: {e}")
        
        # Standard temperature
        try:
            standard = model.generate(intent, temperature=0.5, num_candidates=1)
            print(f"  Standard (T=0.5): {' '.join(standard[0])}")
        except Exception as e:
            print(f"  Standard failed: {e}")
    
    print("\n" + "="*60)
    print("TESTING NOVEL INTENT (NOT IN TRAINING)")
    print("="*60)
    
    # Test on a novel intent not in training data
    novel_intent = {
        "kv_in": ["115kv"],  # Different voltage
        "kv_out": ["13.8kv"], # Single output
        "num_transformers": "1",
        "rated_MVA": "50mva",  # Different MVA
        "percentZ": "6percentZ",
        "hv_interrupting_kA": "20kA",
        "feeder_thermal_A": "600A",
        "reliability_target": "N_1"
    }
    
    print(f"Novel intent: {novel_intent}")
    
    # Test multiple generations
    try:
        candidates = model.generate(novel_intent, temperature=0.3, num_candidates=3)
        for j, candidate in enumerate(candidates):
            print(f"\nCandidate {j+1}: {' '.join(candidate)}")
            
            # Validate
            from src.specgen.grammar_enforcer import GrammarEnforcer
            from src.specgen.vocabulary import ActionVocabulary
            
            vocab = ActionVocabulary()
            enforcer = GrammarEnforcer(vocab)
            
            full_seq = ["<START>"] + candidate + ["<END>"]
            is_valid, error = enforcer.validate_sequence(full_seq)
            print(f"  Validation: {'✅ VALID' if is_valid else f'❌ {error}'}")
    except Exception as e:
        print(f"Novel generation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Summary: Overfitting test complete!")

if __name__ == "__main__":
    test_generation_after_overfitting()
