#!/usr/bin/env python3
"""
Show successful intent-wise examples from the model validation results
"""

import json
import sys

def format_action_sequence(sequence):
    """Format action sequence for readable display"""
    formatted = []
    i = 0
    indent = "  "
    
    while i < len(sequence):
        token = sequence[i]
        
        if token == "ADD_BUS":
            if i + 4 < len(sequence):
                formatted.append(f"{indent}ADD_BUS {sequence[i+1]} {sequence[i+2]} {sequence[i+3]} {sequence[i+4]}")
                i += 5
            else:
                formatted.append(f"{indent}{token}")
                i += 1
                
        elif token == "ADD_TRANSFORMER":
            if i + 5 < len(sequence):
                formatted.append(f"{indent}ADD_TRANSFORMER {sequence[i+1]} {sequence[i+2]} {sequence[i+3]} {sequence[i+4]} {sequence[i+5]}")
                i += 6
            else:
                formatted.append(f"{indent}{token}")
                i += 1
                
        elif token == "ADD_BAY":
            if i + 4 < len(sequence):
                formatted.append(f"{indent}ADD_BAY {sequence[i+1]} {sequence[i+2]} {sequence[i+3]} {sequence[i+4]}")
                i += 5
            else:
                formatted.append(f"{indent}{token}")
                i += 1
                
        elif token == "APPEND_STEP":
            if i + 2 < len(sequence):
                formatted.append(f"{indent}  APPEND_STEP {sequence[i+1]} {sequence[i+2]}")
                i += 3
            else:
                formatted.append(f"{indent}{token}")
                i += 1
                
        elif token in ["CONNECT_BUS", "CONNECT_TX_HV", "CONNECT_TX_LV"]:
            if i + 2 < len(sequence):
                formatted.append(f"{indent}{token} {sequence[i+1]} {sequence[i+2]}")
                i += 3
            else:
                formatted.append(f"{indent}{token}")
                i += 1
                
        else:
            formatted.append(f"{indent}{token}")
            i += 1
    
    return formatted

def show_successful_examples():
    """Display the most successful examples"""
    
    try:
        with open('model_validation_results.json', 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print("❌ model_validation_results.json not found. Please run model_validator.py first.")
        return
    
    print("🎯 SUCCESSFUL INTENT-WISE EXAMPLES FROM TRAINED MODEL")
    print("=" * 80)
    
    # Filter for the most successful examples (perfect alignment score)
    perfect_examples = [
        r for r in results["detailed_results"] 
        if r["alignment"]["overall_score"] >= 1.0 and r["execution_successful"]
    ]
    
    # Also show some good examples (high alignment)
    good_examples = [
        r for r in results["detailed_results"] 
        if r["alignment"]["overall_score"] >= 0.80 and r["execution_successful"] and r not in perfect_examples
    ]
    
    print(f"\n✅ PERFECT EXAMPLES (100% Intent Alignment): {len(perfect_examples)} found")
    print("-" * 60)
    
    for i, example in enumerate(perfect_examples[:3], 1):  # Show first 3 perfect examples
        intent = example["intent"]
        
        print(f"\n🔸 EXAMPLE {i}: {intent['description'].upper()}")
        print(f"Intent Specification:")
        print(f"  • HV Side: {intent['from_kv']}kV")
        print(f"  • LV Side: {intent['to_kv']}kV") 
        print(f"  • Capacity: {intent['rating_mva']}MVA")
        print(f"  • HV Scheme: {intent['scheme_hv']}")
        print(f"  • LV Feeders: {intent['lv_feeders']}")
        print(f"  • Protection: {intent['protection']}")
        print(f"  • Design Philosophy: {example['philosophy']}")
        
        print(f"\nGenerated Action Sequence ({example['sequence_length']} tokens):")
        formatted_sequence = format_action_sequence(example["sequence"])
        for action in formatted_sequence:
            print(action)
        
        # Show final state
        final_state = example["final_state"]
        print(f"\nFinal Substation Configuration:")
        print(f"  • Buses: {final_state['buses']}")
        print(f"  • Transformers: {final_state['transformers']}")
        print(f"  • Bays: {final_state['bays']}")
        print(f"  • Connections: {final_state['connections']}")
        
        # Show alignment details
        alignment = example["alignment"]
        print(f"\nAlignment Verification:")
        print(f"  • HV Voltage Match: ✅" if alignment["hv_voltage_exact"] else f"  • HV Voltage Match: ❌")
        print(f"  • LV Voltage Match: ✅" if alignment["lv_voltage_exact"] else f"  • LV Voltage Match: ❌")
        print(f"  • Capacity Match: ✅" if alignment["capacity_match"] else f"  • Capacity Match: ❌")
        print(f"  • Feeder Count: ✅" if alignment["feeder_count_exact"] else f"  • Feeder Count: ❌")
        print(f"  • Protection Level: ✅" if alignment["protection_appropriate"] else f"  • Protection Level: ❌")
        print(f"  • Bus Scheme: ✅" if alignment["scheme_match"] else f"  • Bus Scheme: ❌")
        print(f"  📊 Overall Score: {alignment['overall_score']:.2f}")
        
        if i < len(perfect_examples):
            print("\n" + "-" * 60)
    
    if good_examples:
        print(f"\n\n⭐ HIGH-QUALITY EXAMPLES (80%+ Intent Alignment): {len(good_examples)} found")
        print("-" * 60)
        
        for i, example in enumerate(good_examples[:2], 1):  # Show 2 good examples
            intent = example["intent"]
            
            print(f"\n🔹 EXAMPLE {len(perfect_examples) + i}: {intent['description'].upper()}")
            print(f"Intent: {intent['from_kv']}kV → {intent['to_kv']}kV, {intent['rating_mva']}MVA ({example['philosophy']} design)")
            
            print(f"\nGenerated Action Sequence ({example['sequence_length']} tokens):")
            formatted_sequence = format_action_sequence(example["sequence"])
            for action in formatted_sequence[:15]:  # Show first 15 actions
                print(action)
            if len(formatted_sequence) > 15:
                print(f"  ... and {len(formatted_sequence) - 15} more actions")
            
            alignment = example["alignment"]
            print(f"\n📊 Alignment Score: {alignment['overall_score']:.2f}")
            if alignment.get("issues"):
                print(f"Minor Issues: {len(alignment['issues'])}")
                for issue in alignment["issues"][:2]:
                    print(f"  • {issue}")
    
    # Summary statistics
    print(f"\n\n📈 SUMMARY STATISTICS")
    print("-" * 40)
    print(f"Total Tests: {results['total_tests']}")
    print(f"Perfect Examples (100%): {len(perfect_examples)}")
    print(f"High-Quality Examples (80%+): {len(good_examples)}")
    print(f"Syntax Valid: {results['syntax_valid']} ({results['syntax_valid']/results['total_tests']:.1%})")
    print(f"Semantic Valid: {results['semantic_valid']} ({results['semantic_valid']/results['total_tests']:.1%})")
    print(f"Intent Aligned (85%+): {results['intent_aligned']} ({results['intent_aligned']/results['total_tests']:.1%})")
    
    # Show design philosophy performance
    philosophy_stats = {}
    for result in results["detailed_results"]:
        phil = result["philosophy"]
        if phil not in philosophy_stats:
            philosophy_stats[phil] = {"total": 0, "perfect": 0, "score_sum": 0}
        philosophy_stats[phil]["total"] += 1
        philosophy_stats[phil]["score_sum"] += result["alignment"]["overall_score"]
        if result["alignment"]["overall_score"] >= 1.0:
            philosophy_stats[phil]["perfect"] += 1
    
    print(f"\n🎨 DESIGN PHILOSOPHY PERFORMANCE:")
    for phil, stats in philosophy_stats.items():
        avg_score = stats["score_sum"] / stats["total"]
        perfect_rate = stats["perfect"] / stats["total"]
        print(f"  • {phil.capitalize()}: {avg_score:.2f} avg score, {perfect_rate:.1%} perfect")

if __name__ == "__main__":
    show_successful_examples()
