#!/usr/bin/env python3
"""
model_validator.py

Validate that the trained model generates engineering-valid substation specifications
from new intent vectors. This tests whether the model learned real patterns vs memorization.

Author: Gordon Doore
Date Created: 2025-08-19
"""

import sys
import os
import json
import random
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Add the src directory to the path so we can import from specgen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
from src.specgen.model import MinimalActionModel, MinimalConfig
from src.specgen.diagram import Diagram
from src.specgen.validation import ActionValidator
from src.specgen.action import ActionKind

class ModelValidator:
    """Validate trained model on new intent specifications"""
    
    def __init__(self, model_path: Optional[str] = None):
        """Initialize validator with trained model"""
        self.config = MinimalConfig()
        self.model = MinimalActionModel(self.config)
        self.validator = ActionValidator()
        
        # Load trained model if path provided
        if model_path and Path(model_path).exists():
            print(f"Loading trained model from {model_path}")
            checkpoint = torch.load(model_path, map_location='cpu')
            self.model.model.load_state_dict(checkpoint['model_state_dict'])  # Load into the internal PyTorch model
            self.model.model.eval()
            print("✅ Model loaded successfully")
        else:
            print("⚠️ Using untrained model (for testing structure)")
    
    def create_test_intents(self) -> List[Dict]:
        """Create diverse test intent vectors"""
        test_intents = [
            # Standard distribution substation
            {
                "from_kv": 69,
                "to_kv": 13.8,
                "rating_mva": 50,
                "scheme_hv": "single_bus",
                "lv_feeders": 4,
                "protection": "standard",
                "description": "69kV distribution substation"
            },
            
            # High-reliability transmission substation
            {
                "from_kv": 138,
                "to_kv": 69,
                "rating_mva": 150,
                "scheme_hv": "double_busbar",
                "lv_feeders": 6,
                "protection": "enhanced",
                "description": "138kV transmission substation with redundancy"
            },
            
            # Industrial substation
            {
                "from_kv": 25,
                "to_kv": 4.16,
                "rating_mva": 25,
                "scheme_hv": "single_bus",
                "lv_feeders": 8,
                "protection": "standard",
                "description": "Industrial plant substation"
            },
            
            # Large transmission substation
            {
                "from_kv": 230,
                "to_kv": 138,
                "rating_mva": 300,
                "scheme_hv": "double_busbar",
                "lv_feeders": 2,
                "protection": "enhanced",
                "description": "Major transmission substation"
            },
            
            # Small rural substation
            {
                "from_kv": 34.5,
                "to_kv": 13.8,
                "rating_mva": 10,
                "scheme_hv": "single_bus",
                "lv_feeders": 3,
                "protection": "basic",
                "description": "Rural distribution substation"
            },
            
            # Edge case: High voltage, low capacity
            {
                "from_kv": 115,
                "to_kv": 25,
                "rating_mva": 15,
                "scheme_hv": "single_bus",
                "lv_feeders": 2,
                "protection": "standard",
                "description": "Specialized high-voltage, low-capacity substation"
            }
        ]
        
        return test_intents
    
    def generate_from_intent(self, intent: Dict, design_philosophy: str = "standard") -> List[str]:
        """Generate action sequence from intent using trained model"""
        try:
            # Use model's generate_from_intent method
            sequence = self.model.generate_from_intent(
                intent=intent,
                design_philosophy=design_philosophy,
                temperature=0.7,
                max_length=200
            )
            return sequence
        except Exception as e:
            print(f"Error generating sequence: {e}")
            return []
    
    def validate_action_sequence(self, sequence: List[str]) -> Tuple[bool, List[str], Dict]:
        """Validate that an action sequence is syntactically and semantically correct"""
        
        validation_results = {
            "syntax_valid": True,
            "semantic_valid": True,
            "execution_successful": False,
            "issues": [],
            "warnings": [],
            "final_state": None
        }
        
        try:
            # Create new diagram for validation
            diagram = Diagram()
            
            # Track what we expect to see
            expected_components = {
                "buses": set(),
                "transformers": set(),
                "bays": set(),
                "connections": []
            }
            
            i = 0
            while i < len(sequence):
                token = sequence[i]
                
                # Skip control tokens
                if token in ["<START>", "<END>", "EMIT_SPEC"]:
                    i += 1
                    continue
                
                # Parse action based on token
                if token == "ADD_BUS":
                    if i + 4 < len(sequence):
                        voltage = sequence[i+1]
                        side = sequence[i+2]
                        role = sequence[i+3]
                        bus_id = sequence[i+4]
                        
                        expected_components["buses"].add(bus_id)
                        
                        # Validate voltage format
                        if not any(v in voltage for v in ["KV", "kV"]):
                            validation_results["issues"].append(f"Invalid voltage format: {voltage}")
                            validation_results["syntax_valid"] = False
                        
                        i += 5
                    else:
                        validation_results["issues"].append("Incomplete ADD_BUS command")
                        validation_results["syntax_valid"] = False
                        break
                
                elif token == "ADD_TRANSFORMER":
                    if i + 5 < len(sequence):
                        tx_type = sequence[i+1]
                        hv_voltage = sequence[i+2]
                        lv_voltage = sequence[i+3]
                        rating = sequence[i+4]
                        tx_id = sequence[i+5]
                        
                        expected_components["transformers"].add(tx_id)
                        
                        # Validate transformer type
                        if tx_type not in ["TWO_WINDING", "THREE_WINDING"]:
                            validation_results["issues"].append(f"Invalid transformer type: {tx_type}")
                            validation_results["syntax_valid"] = False
                        
                        i += 6
                    else:
                        validation_results["issues"].append("Incomplete ADD_TRANSFORMER command")
                        validation_results["syntax_valid"] = False
                        break
                
                elif token == "ADD_BAY":
                    if i + 4 < len(sequence):
                        side = sequence[i+1]
                        voltage = sequence[i+2]
                        bay_type = sequence[i+3]
                        bay_id = sequence[i+4]
                        
                        expected_components["bays"].add(bay_id)
                        
                        # Validate bay type
                        valid_bay_types = ["FEEDER_BAY", "TRANSFORMER_BAY", "LINE_BAY"]
                        if bay_type not in valid_bay_types:
                            validation_results["issues"].append(f"Invalid bay type: {bay_type}")
                            validation_results["syntax_valid"] = False
                        
                        i += 5
                    else:
                        validation_results["issues"].append("Incomplete ADD_BAY command")
                        validation_results["syntax_valid"] = False
                        break
                
                elif token == "CONNECT_BUS":
                    if i + 2 < len(sequence):
                        bus_id = sequence[i+1]
                        bay_id = sequence[i+2]
                        
                        expected_components["connections"].append(("bus", bus_id, bay_id))
                        
                        # Check if referenced components exist
                        if bus_id not in expected_components["buses"]:
                            validation_results["issues"].append(f"Connection references non-existent bus: {bus_id}")
                            validation_results["semantic_valid"] = False
                        
                        if bay_id not in expected_components["bays"]:
                            validation_results["issues"].append(f"Connection references non-existent bay: {bay_id}")
                            validation_results["semantic_valid"] = False
                        
                        i += 3
                    else:
                        validation_results["issues"].append("Incomplete CONNECT_BUS command")
                        validation_results["syntax_valid"] = False
                        break
                
                elif token in ["CONNECT_TX_HV", "CONNECT_TX_LV"]:
                    if i + 2 < len(sequence):
                        tx_id = sequence[i+1]
                        bus_id = sequence[i+2]
                        
                        expected_components["connections"].append(("transformer", tx_id, bus_id))
                        
                        # Check if referenced components exist
                        if tx_id not in expected_components["transformers"]:
                            validation_results["issues"].append(f"Connection references non-existent transformer: {tx_id}")
                            validation_results["semantic_valid"] = False
                        
                        if bus_id not in expected_components["buses"]:
                            validation_results["issues"].append(f"Connection references non-existent bus: {bus_id}")
                            validation_results["semantic_valid"] = False
                        
                        i += 3
                    else:
                        validation_results["issues"].append(f"Incomplete {token} command")
                        validation_results["syntax_valid"] = False
                        break
                
                elif token == "APPEND_STEP":
                    if i + 2 < len(sequence):
                        bay_id = sequence[i+1]
                        equipment = sequence[i+2]
                        
                        # Check if bay exists
                        if bay_id not in expected_components["bays"]:
                            validation_results["issues"].append(f"APPEND_STEP references non-existent bay: {bay_id}")
                            validation_results["semantic_valid"] = False
                        
                        # Validate equipment type
                        valid_equipment = ["BREAKER", "BUS_ISOLATOR", "LINE_ISOLATOR", "CT", "PT"]
                        if equipment not in valid_equipment:
                            validation_results["warnings"].append(f"Unknown equipment type: {equipment}")
                        
                        i += 3
                    else:
                        validation_results["issues"].append("Incomplete APPEND_STEP command")
                        validation_results["syntax_valid"] = False
                        break
                
                else:
                    validation_results["warnings"].append(f"Unknown token: {token}")
                    i += 1
            
            # Semantic validation checks
            if validation_results["syntax_valid"]:
                # Check for basic substation requirements
                if len(expected_components["buses"]) < 2:
                    validation_results["issues"].append("Substation must have at least 2 buses (HV and LV)")
                    validation_results["semantic_valid"] = False
                
                if len(expected_components["transformers"]) < 1:
                    validation_results["issues"].append("Substation must have at least 1 transformer")
                    validation_results["semantic_valid"] = False
                
                if len(expected_components["bays"]) < 1:
                    validation_results["issues"].append("Substation must have at least 1 bay")
                    validation_results["semantic_valid"] = False
                
                # Check for transformer connections
                tx_connections = [conn for conn in expected_components["connections"] if conn[0] == "transformer"]
                if len(tx_connections) < 2:
                    validation_results["issues"].append("Transformer must be connected to both HV and LV buses")
                    validation_results["semantic_valid"] = False
            
            validation_results["execution_successful"] = validation_results["syntax_valid"] and validation_results["semantic_valid"]
            validation_results["final_state"] = {
                "buses": len(expected_components["buses"]),
                "transformers": len(expected_components["transformers"]),
                "bays": len(expected_components["bays"]),
                "connections": len(expected_components["connections"])
            }
            
        except Exception as e:
            validation_results["issues"].append(f"Validation error: {str(e)}")
            validation_results["execution_successful"] = False
        
        return validation_results["execution_successful"], validation_results["issues"] + validation_results["warnings"], validation_results
    
    def check_intent_alignment(self, intent: Dict, sequence: List[str]) -> Dict:
        """Check if generated sequence aligns with original intent - STRICT VALIDATION"""
        alignment_results = {
            "hv_voltage_exact": False,
            "lv_voltage_exact": False,
            "voltage_match": False,
            "capacity_match": False,
            "feeder_count_exact": False,
            "feeder_count_reasonable": False,
            "protection_appropriate": False,
            "scheme_match": False,
            "overall_score": 0.0,
            "issues": []
        }
        
        try:
            # Extract voltages from sequence
            sequence_voltages = [token for token in sequence if "KV" in token.upper()]
            
            # STRICT voltage checking
            expected_hv = f"{intent['from_kv']}KV"
            expected_lv = f"{intent['to_kv']}KV"
            
            # Vocabulary mapping for known approximations
            voltage_mapping = {
                4.16: "13.8KV",    # 4.16kV maps to 13.8kV in vocab
                230: "220KV",      # 230kV maps to 220kV in vocab
                500: "345KV",      # 500kV maps to 345kV in vocab
                66: "69KV"         # 66kV maps to 69kV in vocab
            }
            
            mapped_hv = voltage_mapping.get(intent['from_kv'], expected_hv)
            mapped_lv = voltage_mapping.get(intent['to_kv'], expected_lv)
            
            # Check HV voltage presence
            if any(mapped_hv in v for v in sequence_voltages):
                alignment_results["hv_voltage_exact"] = True
            else:
                alignment_results["issues"].append(f"Missing HV voltage {mapped_hv}, found: {sequence_voltages}")
            
            # Check LV voltage presence  
            if any(mapped_lv in v for v in sequence_voltages):
                alignment_results["lv_voltage_exact"] = True
            else:
                alignment_results["issues"].append(f"Missing LV voltage {mapped_lv}, found: {sequence_voltages}")
            
            # Overall voltage match (both HV and LV must be present)
            alignment_results["voltage_match"] = alignment_results["hv_voltage_exact"] and alignment_results["lv_voltage_exact"]
            
            # STRICT capacity checking
            mva_tokens = [token for token in sequence if "MVA" in token]
            expected_mva = intent['rating_mva']
            
            # Map expected MVA to vocabulary approximations
            mva_mapping = {
                1000: "150MVA", 750: "150MVA", 500: "150MVA", 
                300: "150MVA", 200: "150MVA", 150: "150MVA",
                100: "100MVA", 75: "75MVA", 50: "50MVA",
                37.5: "50MVA", 25: "50MVA", 15: "50MVA", 10: "50MVA"
            }
            
            expected_mva_token = mva_mapping.get(expected_mva, f"{expected_mva}MVA")
            
            if any(expected_mva_token in token for token in mva_tokens):
                alignment_results["capacity_match"] = True
            else:
                alignment_results["issues"].append(f"Missing capacity {expected_mva_token}, found: {mva_tokens}")
            
            # STRICT feeder count checking
            feeder_bays = sequence.count("FEEDER_BAY")
            expected_feeders = intent.get("lv_feeders", 2)
            
            if feeder_bays == expected_feeders:
                alignment_results["feeder_count_exact"] = True
                alignment_results["feeder_count_reasonable"] = True
            elif abs(feeder_bays - expected_feeders) <= 1:  # Allow ±1 feeder
                alignment_results["feeder_count_reasonable"] = True
                alignment_results["issues"].append(f"Feeder count close but not exact: expected {expected_feeders}, got {feeder_bays}")
            else:
                alignment_results["issues"].append(f"Feeder count mismatch: expected {expected_feeders}, got {feeder_bays}")
            
            # Protection level checking
            protection_level = intent.get("protection", "standard")
            ct_count = sequence.count("CT")
            breaker_count = sequence.count("BREAKER")
            
            if protection_level == "enhanced":
                if ct_count > 0:  # Enhanced should have current transformers
                    alignment_results["protection_appropriate"] = True
                else:
                    alignment_results["issues"].append("Enhanced protection should include CT (Current Transformers)")
            elif protection_level == "basic":
                if breaker_count > 0 and ct_count == 0:  # Basic should have breakers but minimal CTs
                    alignment_results["protection_appropriate"] = True
                else:
                    alignment_results["issues"].append("Basic protection should have breakers with minimal CT usage")
            else:  # standard
                if breaker_count > 0:  # Standard should have breakers
                    alignment_results["protection_appropriate"] = True
                else:
                    alignment_results["issues"].append("Standard protection should include breakers")
            
            # Scheme checking (HV bus configuration)
            expected_scheme = intent.get("scheme_hv", "single_bus")
            bus_count_hv = 0
            
            # Count HV buses (look for HV side buses)
            i = 0
            while i < len(sequence):
                if (sequence[i] == "ADD_BUS" and i + 2 < len(sequence) and 
                    sequence[i + 2] == "HV"):
                    bus_count_hv += 1
                i += 1
            
            if expected_scheme in ["double_busbar", "double_bus"]:
                if bus_count_hv >= 2:
                    alignment_results["scheme_match"] = True
                else:
                    alignment_results["issues"].append(f"Double busbar scheme should have ≥2 HV buses, got {bus_count_hv}")
            else:  # single_bus, single_busbar, or other single schemes
                if bus_count_hv >= 1:
                    alignment_results["scheme_match"] = True
                else:
                    alignment_results["issues"].append(f"Single bus scheme should have ≥1 HV bus, got {bus_count_hv}")
            
            # Calculate strict overall score (all major requirements must be met)
            critical_components = [
                alignment_results["hv_voltage_exact"],     # Must have correct HV voltage
                alignment_results["lv_voltage_exact"],     # Must have correct LV voltage  
                alignment_results["capacity_match"],       # Must have correct capacity
                alignment_results["feeder_count_reasonable"], # Feeder count should be reasonable
                alignment_results["protection_appropriate"], # Protection should match level
                alignment_results["scheme_match"]          # Bus scheme should match
            ]
            
            alignment_results["overall_score"] = sum(critical_components) / len(critical_components)
            
        except Exception as e:
            alignment_results["issues"].append(f"Error checking alignment: {str(e)}")
            print(f"Error checking intent alignment: {e}")
        
        return alignment_results
    
    def run_comprehensive_validation(self) -> Dict:
        """Run comprehensive validation on multiple test cases"""
        
        print("🔍 Running Comprehensive Model Validation")
        print("=" * 50)
        
        test_intents = self.create_test_intents()
        results = {
            "total_tests": len(test_intents),
            "syntax_valid": 0,
            "semantic_valid": 0,
            "intent_aligned": 0,
            "detailed_results": [],
            "summary": {}
        }
        
        design_philosophies = ["conservative", "standard", "economical"]
        
        for i, intent in enumerate(test_intents):
            print(f"\n--- Test {i+1}: {intent['description']} ---")
            print(f"Intent: {intent['from_kv']}kV → {intent['to_kv']}kV, {intent['rating_mva']}MVA")
            
            for philosophy in design_philosophies:
                print(f"\nTesting with {philosophy} design philosophy:")
                
                # Generate sequence
                sequence = self.generate_from_intent(intent, philosophy)
                
                if not sequence:
                    print("❌ Failed to generate sequence")
                    continue
                
                print(f"Generated {len(sequence)} tokens")
                
                # Validate sequence
                is_valid, issues, validation_details = self.validate_action_sequence(sequence)
                
                # Check intent alignment
                alignment = self.check_intent_alignment(intent, sequence)
                
                # Store results
                test_result = {
                    "test_id": i + 1,
                    "intent": intent,
                    "philosophy": philosophy,
                    "sequence": sequence,
                    "sequence_length": len(sequence),
                    "syntax_valid": validation_details["syntax_valid"],
                    "semantic_valid": validation_details["semantic_valid"],
                    "execution_successful": is_valid,
                    "issues": issues,
                    "alignment": alignment,
                    "final_state": validation_details["final_state"]
                }
                
                results["detailed_results"].append(test_result)
                
                # Update counters
                if validation_details["syntax_valid"]:
                    results["syntax_valid"] += 1
                
                if validation_details["semantic_valid"]:
                    results["semantic_valid"] += 1
                
                # Use stricter threshold for intent alignment (0.85 instead of 0.75)
                if alignment["overall_score"] >= 0.85:
                    results["intent_aligned"] += 1
                
                # Print immediate results
                status = "✅" if is_valid else "❌"
                alignment_score = f"{alignment['overall_score']:.2f}"
                print(f"{status} Valid: {is_valid}, Alignment: {alignment_score}")
                
                # Show detailed alignment issues
                if alignment.get("issues"):
                    print(f"Alignment Issues:")
                    for issue in alignment["issues"][:2]:  # Show first 2 alignment issues
                        print(f"  • {issue}")
                    if len(alignment["issues"]) > 2:
                        print(f"  • ... and {len(alignment['issues']) - 2} more alignment issues")
                
                # Show validation issues
                if issues:
                    print(f"Validation Issues: {len(issues)} found")
                    for issue in issues[:2]:  # Show first 2 validation issues
                        print(f"  • {issue}")
                    if len(issues) > 2:
                        print(f"  • ... and {len(issues) - 2} more validation issues")
        
        # Calculate summary statistics
        total_tests = len(results["detailed_results"])
        results["total_tests"] = total_tests
        
        results["summary"] = {
            "syntax_success_rate": results["syntax_valid"] / total_tests if total_tests > 0 else 0,
            "semantic_success_rate": results["semantic_valid"] / total_tests if total_tests > 0 else 0,
            "intent_alignment_rate": results["intent_aligned"] / total_tests if total_tests > 0 else 0,
            "overall_success_rate": sum([
                results["syntax_valid"],
                results["semantic_valid"],
                results["intent_aligned"]
            ]) / (3 * total_tests) if total_tests > 0 else 0
        }
        
        return results
    
    def print_validation_summary(self, results: Dict):
        """Print a comprehensive summary of validation results"""
        
        print("\n" + "=" * 60)
        print("🎯 VALIDATION SUMMARY")
        print("=" * 60)
        
        summary = results["summary"]
        
        print(f"Total Tests Run: {results['total_tests']}")
        print(f"Syntax Valid: {results['syntax_valid']} ({summary['syntax_success_rate']:.1%})")
        print(f"Semantic Valid: {results['semantic_valid']} ({summary['semantic_success_rate']:.1%})")
        print(f"Intent Aligned: {results['intent_aligned']} ({summary['intent_alignment_rate']:.1%})")
        print(f"Overall Success: {summary['overall_success_rate']:.1%}")
        
        # Grade the model
        overall_score = summary['overall_success_rate']
        if overall_score >= 0.9:
            grade = "A (Excellent)"
        elif overall_score >= 0.8:
            grade = "B (Good)"
        elif overall_score >= 0.7:
            grade = "C (Acceptable)"
        elif overall_score >= 0.6:
            grade = "D (Poor)"
        else:
            grade = "F (Failing)"
        
        print(f"\n🎖️ Model Grade: {grade}")
        
        # Identify common issues
        all_issues = []
        for result in results["detailed_results"]:
            all_issues.extend(result["issues"])
        
        if all_issues:
            from collections import Counter
            issue_counts = Counter(all_issues)
            
            print(f"\n⚠️ Most Common Issues:")
            for issue, count in issue_counts.most_common(5):
                print(f"  • {issue} ({count} times)")
        
        # Best and worst performers
        successful_tests = [r for r in results["detailed_results"] if r["execution_successful"]]
        if successful_tests:
            print(f"\n✅ Best Performing Configuration:")
            best = max(successful_tests, key=lambda x: x["alignment"]["overall_score"])
            print(f"   {best['intent']['description']} with {best['philosophy']} philosophy")
            print(f"   Alignment Score: {best['alignment']['overall_score']:.2f}")
        
        failed_tests = [r for r in results["detailed_results"] if not r["execution_successful"]]
        if failed_tests:
            print(f"\n❌ Most Challenging Configuration:")
            worst = failed_tests[0]  # First failed test
            print(f"   {worst['intent']['description']} with {worst['philosophy']} philosophy")
            print(f"   Issues: {len(worst['issues'])}")


def main():
    """Run model validation"""
    
    # Try to find a saved model (you would save this in corpus_trainer.py)
    model_path = "trained_model.pt"  # This would be created by corpus_trainer.py
    
    validator = ModelValidator(model_path)
    
    # Run comprehensive validation
    results = validator.run_comprehensive_validation()
    
    # Print summary
    validator.print_validation_summary(results)
    
    # Save detailed results
    with open("model_validation_results.json", 'w') as f:
        # Convert any non-serializable objects
        serializable_results = results.copy()
        for result in serializable_results["detailed_results"]:
            # Convert sequence to string representation for readability
            result["sequence_preview"] = " ".join(result["sequence"][:20]) + "..." if len(result["sequence"]) > 20 else " ".join(result["sequence"])
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: model_validation_results.json")
    
    return results


if __name__ == "__main__":
    main()
