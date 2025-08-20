#!/usr/bin/env python3
"""
Complete example showing trained model generating engineering-valid substation specifications
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from src.specgen.model import MinimalActionModel, MinimalConfig

def demonstrate_complete_example():
    """Show a complete example with engineering analysis"""
    
    print("🏭 COMPLETE ENGINEERING EXAMPLE: INDUSTRIAL SUBSTATION DESIGN")
    print("=" * 80)
    
    # Load trained model
    config = MinimalConfig()
    model = MinimalActionModel(config)
    
    try:
        checkpoint = torch.load('trained_model.pt', map_location='cpu')
        model.model.load_state_dict(checkpoint['model_state_dict'])
        model.model.eval()
        print("✅ Trained model loaded successfully")
    except Exception as e:
        print(f"❌ Could not load trained model: {e}")
        return
    
    # Real-world industrial substation specification
    industrial_intent = {
        "from_kv": 138,
        "to_kv": 13.8,
        "rating_mva": 100,
        "scheme_hv": "double_busbar",
        "lv_feeders": 4,
        "protection": "enhanced",
        "description": "138kV/13.8kV Industrial Substation with Redundancy"
    }
    
    print(f"\n📋 ENGINEERING SPECIFICATION:")
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Project: {industrial_intent['description']}")
    print(f"Primary Voltage: {industrial_intent['from_kv']}kV")
    print(f"Secondary Voltage: {industrial_intent['to_kv']}kV")
    print(f"Transformer Rating: {industrial_intent['rating_mva']}MVA")
    print(f"HV Bus Configuration: {industrial_intent['scheme_hv']}")
    print(f"Number of LV Feeders: {industrial_intent['lv_feeders']}")
    print(f"Protection Level: {industrial_intent['protection']}")
    
    # Generate with conservative approach for industrial application
    print(f"\n🎨 GENERATING CONSERVATIVE DESIGN (High Reliability):")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    try:
        sequence = model.generate_from_intent(
            intent=industrial_intent,
            design_philosophy="conservative",
            temperature=0.5,  # Balanced temperature for reasonable variation
            max_length=200
        )
        
        if not sequence:
            print("❌ Failed to generate sequence")
            return
        
        print(f"Generated complete specification with {len(sequence)} action tokens")
        
        # Analyze the generated sequence
        print(f"\n🔍 ENGINEERING ANALYSIS:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        # Count components
        components = analyze_components(sequence)
        
        print(f"🔧 Component Summary:")
        print(f"  • Buses: {components['buses']} ({components['hv_buses']} HV, {components['lv_buses']} LV)")
        print(f"  • Transformers: {components['transformers']}")
        print(f"  • Bays: {components['bays']} ({components['tx_bays']} transformer, {components['feeder_bays']} feeder)")
        print(f"  • Protection Devices: {components['protection_devices']}")
        print(f"  • Connections: {components['connections']}")
        
        # Voltage analysis
        voltages = extract_voltages(sequence)
        print(f"\n⚡ Voltage Configuration:")
        print(f"  • HV Side: {', '.join(voltages['hv'])}")
        print(f"  • LV Side: {', '.join(voltages['lv'])}")
        
        # Protection analysis  
        protection = analyze_protection(sequence)
        print(f"\n🛡️ Protection System:")
        print(f"  • Circuit Breakers: {protection['breakers']}")
        print(f"  • Current Transformers: {protection['cts']}")
        print(f"  • Bus Isolators: {protection['bus_isolators']}")
        print(f"  • Line Isolators: {protection['line_isolators']}")
        
        # Engineering validation
        print(f"\n✅ ENGINEERING VALIDATION:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        validation_results = validate_engineering_design(sequence, industrial_intent)
        
        for check, result in validation_results.items():
            status = "✅" if result['passed'] else "❌"
            print(f"{status} {check}: {result['message']}")
        
        # Show complete action sequence in readable format
        print(f"\n📜 COMPLETE ACTION SEQUENCE:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        formatted_actions = format_complete_sequence(sequence)
        for i, action in enumerate(formatted_actions, 1):
            print(f"{i:2d}. {action}")
        
        # Grammar verification
        full_sequence = ["<START>"] + sequence + ["<END>"]
        is_valid = model.grammar.is_sequence_complete(full_sequence)
        print(f"\n🔍 Grammar Validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
        # Show design alternatives
        print(f"\n🎨 DESIGN ALTERNATIVES:")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━")
        
        for philosophy in ["standard", "economical"]:
            alt_sequence = model.generate_from_intent(
                intent=industrial_intent,
                design_philosophy=philosophy,
                temperature=0.5
            )
            
            alt_components = analyze_components(alt_sequence)
            cost_factor = estimate_relative_cost(alt_components, components)
            
            print(f"\n  {philosophy.upper()} Design:")
            print(f"    • Components: {alt_components['buses']} buses, {alt_components['bays']} bays")
            print(f"    • Protection: {alt_components['protection_devices']} devices")
            print(f"    • Relative Cost: {cost_factor:.1f}x baseline")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def analyze_components(sequence):
    """Analyze components in the sequence"""
    components = {
        'buses': sequence.count('ADD_BUS'),
        'transformers': sequence.count('ADD_TRANSFORMER'),
        'bays': sequence.count('ADD_BAY'),
        'hv_buses': 0,
        'lv_buses': 0,
        'tx_bays': sequence.count('TRANSFORMER_BAY'),
        'feeder_bays': sequence.count('FEEDER_BAY'),
        'protection_devices': sequence.count('BREAKER') + sequence.count('CT') + 
                              sequence.count('BUS_ISOLATOR') + sequence.count('LINE_ISOLATOR'),
        'connections': sequence.count('CONNECT_BUS') + sequence.count('CONNECT_TX_HV') + 
                       sequence.count('CONNECT_TX_LV')
    }
    
    # Count HV and LV buses
    i = 0
    while i < len(sequence):
        if sequence[i] == 'ADD_BUS' and i + 2 < len(sequence):
            if sequence[i + 2] == 'HV':
                components['hv_buses'] += 1
            elif sequence[i + 2] == 'LV':
                components['lv_buses'] += 1
        i += 1
    
    return components

def extract_voltages(sequence):
    """Extract voltage information"""
    voltages = {'hv': set(), 'lv': set()}
    
    i = 0
    while i < len(sequence):
        if sequence[i] == 'ADD_BUS' and i + 2 < len(sequence):
            voltage = sequence[i + 1]
            side = sequence[i + 2]
            if side == 'HV':
                voltages['hv'].add(voltage)
            elif side == 'LV':
                voltages['lv'].add(voltage)
        i += 1
    
    return {k: sorted(list(v)) for k, v in voltages.items()}

def analyze_protection(sequence):
    """Analyze protection equipment"""
    return {
        'breakers': sequence.count('BREAKER'),
        'cts': sequence.count('CT'),
        'bus_isolators': sequence.count('BUS_ISOLATOR'),
        'line_isolators': sequence.count('LINE_ISOLATOR')
    }

def validate_engineering_design(sequence, intent):
    """Validate engineering aspects of the design"""
    results = {}
    
    # Check basic structure
    has_hv_bus = any('HV' in sequence[i+2] for i in range(len(sequence)-2) if sequence[i] == 'ADD_BUS')
    has_lv_bus = any('LV' in sequence[i+2] for i in range(len(sequence)-2) if sequence[i] == 'ADD_BUS')
    has_transformer = 'ADD_TRANSFORMER' in sequence
    
    results['Basic Structure'] = {
        'passed': has_hv_bus and has_lv_bus and has_transformer,
        'message': 'HV bus, LV bus, and transformer present' if has_hv_bus and has_lv_bus and has_transformer else 'Missing basic components'
    }
    
    # Check feeder count
    feeder_count = sequence.count('FEEDER_BAY')
    expected_feeders = intent.get('lv_feeders', 2)
    
    results['Feeder Count'] = {
        'passed': abs(feeder_count - expected_feeders) <= 1,
        'message': f'{feeder_count} feeders (expected {expected_feeders})'
    }
    
    # Check protection level
    protection_level = intent.get('protection', 'standard')
    ct_count = sequence.count('CT')
    breaker_count = sequence.count('BREAKER')
    
    protection_adequate = True
    protection_msg = f'{breaker_count} breakers, {ct_count} CTs'
    
    if protection_level == 'enhanced' and ct_count < 1:
        protection_adequate = False
        protection_msg += ' (enhanced protection should include CTs)'
    elif protection_level == 'basic' and breaker_count < 1:
        protection_adequate = False
        protection_msg += ' (basic protection requires breakers)'
    
    results['Protection Level'] = {
        'passed': protection_adequate,
        'message': protection_msg
    }
    
    # Check busbar scheme
    scheme = intent.get('scheme_hv', 'single_bus')
    hv_bus_count = sum(1 for i in range(len(sequence)-2) 
                       if sequence[i] == 'ADD_BUS' and sequence[i+2] == 'HV')
    
    scheme_ok = True
    if scheme in ['double_busbar', 'double_bus'] and hv_bus_count < 2:
        scheme_ok = False
    
    results['Bus Scheme'] = {
        'passed': scheme_ok,
        'message': f'{hv_bus_count} HV buses for {scheme} scheme'
    }
    
    return results

def format_complete_sequence(sequence):
    """Format sequence for complete display"""
    formatted = []
    i = 0
    
    while i < len(sequence):
        token = sequence[i]
        
        if token == "ADD_BUS" and i + 4 < len(sequence):
            formatted.append(f"Create {sequence[i+2]} bus {sequence[i+4]} at {sequence[i+1]} ({sequence[i+3]} role)")
            i += 5
        elif token == "ADD_TRANSFORMER" and i + 5 < len(sequence):
            formatted.append(f"Install {sequence[i+1]} transformer {sequence[i+5]}: {sequence[i+2]} → {sequence[i+3]}, {sequence[i+4]}")
            i += 6
        elif token == "ADD_BAY" and i + 4 < len(sequence):
            formatted.append(f"Construct {sequence[i+3]} {sequence[i+4]} on {sequence[i+1]} side ({sequence[i+2]})")
            i += 5
        elif token == "APPEND_STEP" and i + 2 < len(sequence):
            formatted.append(f"  └─ Install {sequence[i+2]} in bay {sequence[i+1]}")
            i += 3
        elif token == "CONNECT_BUS" and i + 2 < len(sequence):
            formatted.append(f"Connect bus {sequence[i+1]} to bay {sequence[i+2]}")
            i += 3
        elif token == "CONNECT_TX_HV" and i + 2 < len(sequence):
            formatted.append(f"Connect transformer {sequence[i+1]} HV side to bus {sequence[i+2]}")
            i += 3
        elif token == "CONNECT_TX_LV" and i + 2 < len(sequence):
            formatted.append(f"Connect transformer {sequence[i+1]} LV side to bus {sequence[i+2]}")
            i += 3
        else:
            formatted.append(token)
            i += 1
    
    return formatted

def estimate_relative_cost(alt_components, base_components):
    """Estimate relative cost based on component counts"""
    # Simple cost model based on major components
    base_cost = (base_components['buses'] * 1.0 + 
                base_components['transformers'] * 5.0 + 
                base_components['bays'] * 2.0 +
                base_components['protection_devices'] * 0.5)
    
    alt_cost = (alt_components['buses'] * 1.0 + 
               alt_components['transformers'] * 5.0 + 
               alt_components['bays'] * 2.0 +
               alt_components['protection_devices'] * 0.5)
    
    return alt_cost / base_cost if base_cost > 0 else 1.0

if __name__ == "__main__":
    demonstrate_complete_example()
