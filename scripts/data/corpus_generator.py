"""
corpus_generator.py - Script to generate synthetic data for training a minimal action model.

This script creates a corpus of action sequences based on predefined intents and parameters.

data format:

{
    "intent": {
      "kv_in": ["230kv"],
      "kv_out": ["13.8kv", "13.8kv", "13.8kv"],
      "num_transformers": "1",
      "rated_MVA": "150mva",
      "percentZ": "8percentZ",
      "hv_interrupting_kA": "31.5kA",
      "feeder_thermal_A": "1200A",
      "reliability_target": "N_1"
    },
    "sequence": [
      "<START>",
      "ADD_BUS", "id=", "bus100", "kv=", "230kv",
      "ADD_BUS", "id=", "bus101", "kv=", "13.8kv",
      "ADD_TRANSFORMER", "id=", "transformer601", "kind=", "TWO_WINDING", 
      "kv_in=", "230kv", "kv_out=", "13.8kv", "tert_kv=", "13.8kv", 
      "rated_MVA=", "150mva", "vector_group=", "Dd0", "percentZ=", "8percentZ",
      "CONNECT", "series=", "[", "bus100", ",", "transformer601", ",", "bus101", "]",
      "EMIT_SPEC",
      "<END>"
    ]
  }

We want to generate a rich corpus with balanced inputs and outputs to train the model effectively.

Author: Gordon Doore
Date Created: 2025-08-22

"""

import json
import random
import os
import sys
from typing import Dict, List, Tuple
from collections import defaultdict

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.specgen.vocabulary import ActionVocabulary
from src.specgen.grammar_enforcer import GrammarEnforcer

class SubstationArchitecture:
    """Defines different substation architecture patterns"""
    
    ARCHITECTURES = {
        "single_transformer_radial": {
            "description": "Single transformer, radial topology",
            "complexity": "low",
            "reliability": ["N_0"],
            "min_transformers": 1,
            "max_transformers": 1,
            "requires_redundancy": False,
            "topology": "radial"
        },
        "dual_transformer_redundant": {
            "description": "Two transformers with redundant connections",
            "complexity": "medium",
            "reliability": ["N_1"],
            "min_transformers": 2,
            "max_transformers": 2,
            "requires_redundancy": True,
            "topology": "redundant"
        },
        "triple_transformer_high_reliability": {
            "description": "Three transformers for high reliability",
            "complexity": "high",
            "reliability": ["N_2", "REDUNDANT"],
            "min_transformers": 3,
            "max_transformers": 3,
            "requires_redundancy": True,
            "topology": "mesh"
        },
        "ring_bus": {
            "description": "Ring bus configuration",
            "complexity": "medium",
            "reliability": ["N_1"],
            "min_transformers": 2,
            "max_transformers": 4,
            "requires_redundancy": True,
            "topology": "ring"
        },
        "double_busbar": {
            "description": "Double busbar with tie breaker",
            "complexity": "high",
            "reliability": ["N_1", "N_2"],
            "min_transformers": 2,
            "max_transformers": 6,
            "requires_redundancy": True,
            "topology": "double_busbar"
        },
        "breaker_and_half": {
            "description": "Breaker and a half scheme",
            "complexity": "high",
            "reliability": ["N_2", "REDUNDANT"],
            "min_transformers": 2,
            "max_transformers": 4,
            "requires_redundancy": True,
            "topology": "breaker_half"
        }
    }

class DataGenerator: 
    def __init__(self, output_file: str):
        self.output_file = output_file
        
        # Initialize vocabulary and grammar for validation
        self.vocab = ActionVocabulary()
        self.grammar = GrammarEnforcer(self.vocab)
        
        # Component ID generators
        # Initialize tracking for component IDs to use sequential numbering
        self.used_ids = set()  # Keep for backward compatibility
        self.component_counters = defaultdict(int)  # Keep for fallback
        self.component_id_counters = defaultdict(int)  # New sequential counters
        
        # Architecture definitions
        self.architectures = SubstationArchitecture.ARCHITECTURES
        
        # Parameter definitions
        self.param_sets = self._define_parameter_sets()
        
        # Generated data storage
        self.generated_data = []
    
    def _define_parameter_sets(self) -> Dict:
        """Define balanced parameter sets for different scenarios - using only vocabulary tokens"""
        # Get available tokens from vocabulary
        token_sets = self.vocab.get_token_sets()
        
        return {
            "voltage_levels": {
                "transmission": {
                    "kv_in": ["345kv", "230kv", "138kv"],
                    "kv_out": ["138kv", "69kv", "25kv", "13.8kv"],
                    "typical_combinations": [
                        (["345kv"], ["138kv", "25kv"]),
                        (["230kv"], ["69kv", "13.8kv"]),
                        (["138kv"], ["25kv", "13.8kv"]),
                    ]
                },
                "distribution": {
                    "kv_in": ["69kv", "46kv", "25kv"],
                    "kv_out": ["13.8kv", "4.16kv"],
                    "typical_combinations": [
                        (["69kv"], ["13.8kv"]),
                        (["46kv"], ["13.8kv", "4.16kv"]),
                        (["25kv"], ["4.16kv"]),
                    ]
                }
            },
            # Use actual vocabulary tokens
            "transformer_ratings": {
                "transmission": ["300mva", "150mva", "100mva", "75mva"],
                "distribution": ["50mva", "25mva", "10mva", "5mva"]
            },
            "impedance_values": list(token_sets["percentz_values"]),  # ['2percentZ', '8percentZ', '4percentZ', '0.5percentZ', '10percentZ', '1percentZ', '6percentZ']
            "breaker_ratings": {
                # Only use kA values that exist in vocabulary
                "345kv": {"interrupting": ["63kA", "50kA", "40kA"], "continuous": ["1200A", "1000A"]},
                "230kv": {"interrupting": ["50kA", "40kA", "31.5kA"], "continuous": ["1200A", "1000A"]},
                "138kv": {"interrupting": ["40kA", "31.5kA", "25kA"], "continuous": ["1000A", "800A"]},
                "115kv": {"interrupting": ["31.5kA", "25kA", "20kA"], "continuous": ["800A", "600A"]},
                "69kv": {"interrupting": ["31.5kA", "25kA", "20kA"], "continuous": ["800A", "600A"]},
                "66kv": {"interrupting": ["25kA", "20kA", "16kA"], "continuous": ["600A", "400A"]},
                "46kv": {"interrupting": ["25kA", "20kA", "16kA"], "continuous": ["600A", "400A"]},
                "34.5kv": {"interrupting": ["20kA", "16kA", "12.5kA"], "continuous": ["400A", "200A"]},
                "25kv": {"interrupting": ["25kA", "20kA", "16kA"], "continuous": ["600A", "400A"]},
                "13.8kv": {"interrupting": ["20kA", "16kA", "12.5kA"], "continuous": ["400A", "200A"]},
                "4.16kv": {"interrupting": ["16kA", "12.5kA", "10kA"], "continuous": ["400A", "200A"]}
            },
            "reliability_targets": list(token_sets["reliability_targets"]),  # ['N_1', 'N_0', 'N_2', 'REDUNDANT', 'MINIMAL']
            "available_voltages": list(token_sets["voltages"]),  # All available voltage tokens
            "available_mva": list(token_sets["mva_values"]),     # All available MVA tokens
            "available_ka": list(token_sets["ka_values"]),       # All available kA tokens
            "available_thermal": [token for token in token_sets["thermal_values"] if token.endswith("A")],  # Only A values
            "transformer_kinds": list(token_sets["transformer_kinds"]),  # ['AUTO', 'THREE_WINDING', 'TWO_WINDING', 'GROUNDING']
            "breaker_kinds": list(token_sets["breaker_kinds"]),            # ['VACUUM', 'SF6', 'OIL', 'AIRBLAST']
            "vector_groups": ["Dd0", "Dyn7", "YNd1", "YNd11", "Yy6"]      # Common transformer vector groups from vocab
        }
    
    def _reset_id_counters(self):
        """Reset ID counters for each new design to ensure sequential numbering starts fresh"""
        self.component_id_counters = defaultdict(int)
        self.used_ids.clear()
    
    def _generate_component_id(self, component_type: str) -> str:
        """Generate component IDs using vocabulary tokens in sequential order"""
        # Map component types to vocabulary token sets
        token_set_map = {
            "bus": "bus_ids",
            "transformer": "transformer_ids", 
            "breaker": "breaker_ids",
            "disconnector": "disconnector_ids",
            "bay": "bay_ids",
            "coupler": "coupler_ids",
            "line": "line_ids"
        }
        
        token_set_name = token_set_map.get(component_type)
        if token_set_name:
            token_sets = self.vocab.get_token_sets()
            if token_set_name in token_sets:
                # Get available IDs and exclude special tokens like 'bus='
                available_ids = [token for token in token_sets[token_set_name] 
                               if not token.endswith('=') and not token.endswith('_id=')]
                if available_ids:
                    # Sort IDs to ensure sequential assignment (bus100, bus101, bus102...)
                    def extract_number(id_token):
                        # Extract number from tokens like 'bus100', 'transformer601', etc.
                        import re
                        match = re.search(r'(\d+)$', id_token)
                        return int(match.group(1)) if match else 0
                    
                    available_ids.sort(key=extract_number)
                    
                    # Track used IDs per component type to ensure sequential assignment
                    if not hasattr(self, 'component_id_counters'):
                        self.component_id_counters = defaultdict(int)
                    
                    # Get the next sequential ID for this component type
                    counter = self.component_id_counters[component_type]
                    if counter < len(available_ids):
                        chosen_id = available_ids[counter]
                        self.component_id_counters[component_type] += 1
                        return chosen_id
                    else:
                        # If we've used all available IDs, wrap around (shouldn't happen normally)
                        chosen_id = available_ids[0]
                        self.component_id_counters[component_type] = 1
                        return chosen_id
        
        # Fallback (shouldn't be needed with proper vocabulary)
        self.component_counters[component_type] += 1
        return f"{component_type}{100 + self.component_counters[component_type]}"
    
    def _select_voltage_combination(self, complexity: str) -> Tuple[List[str], List[str]]:
        """Select voltage combination based on complexity"""
        voltage_sets = self.param_sets["voltage_levels"]
        
        if complexity == "low":
            # Simple single voltage transformation
            combinations = voltage_sets["distribution"]["typical_combinations"]
            kv_in, kv_out = random.choice(combinations)
            return kv_in, kv_out[:1]  # Single output voltage
            
        elif complexity == "medium":
            # Mixed or multiple output voltages
            if random.random() < 0.5:
                # Multiple output voltages at same level
                combinations = voltage_sets["transmission"]["typical_combinations"]
                kv_in, kv_out = random.choice(combinations)
                return kv_in, kv_out
            else:
                # Distribution with multiple feeders
                combinations = voltage_sets["distribution"]["typical_combinations"] 
                kv_in, kv_out = random.choice(combinations)
                return kv_in, kv_out * random.randint(2, 3)  # Multiple feeders
                
        else:  # high complexity
            # Complex multi-level transformations
            if random.random() < 0.3:
                # Multiple input voltages (interconnection)
                tx_combo = random.choice(voltage_sets["transmission"]["typical_combinations"])
                dist_combo = random.choice(voltage_sets["distribution"]["typical_combinations"])
                return tx_combo[0] + dist_combo[0][:1], tx_combo[1] + dist_combo[1]
            else:
                # High voltage with multiple outputs
                kv_in, kv_out = random.choice(voltage_sets["transmission"]["typical_combinations"])
                # Add extra output voltages
                extra_voltages = random.sample(voltage_sets["distribution"]["kv_out"], 
                                             random.randint(1, 2))
                return kv_in, kv_out + extra_voltages
    
    def _select_transformer_parameters(self, kv_in: List[str], kv_out: List[str], 
                                     num_transformers: int, complexity: str) -> Dict:
        """Select appropriate transformer parameters"""
        # Determine if transmission or distribution based on voltage levels
        max_voltage = max([int(v.replace('kv', '').replace('.', '')) 
                          for v in kv_in + kv_out])
        
        # Select MVA from available vocabulary tokens
        if max_voltage >= 138:
            # Prefer larger MVA for transmission
            available_mva = self.param_sets["available_mva"]
            large_mva = [mva for mva in available_mva if float(mva.replace('mva', '')) >= 100]
            mva_options = large_mva if large_mva else available_mva
        else:
            # Prefer smaller MVA for distribution  
            available_mva = self.param_sets["available_mva"]
            small_mva = [mva for mva in available_mva if float(mva.replace('mva', '')) <= 75]
            mva_options = small_mva if small_mva else available_mva
        
        # Select MVA based on complexity
        if complexity == "low":
            rated_mva = random.choice(mva_options[-2:])  # Smaller ratings
        elif complexity == "medium":
            rated_mva = random.choice(mva_options[1:3] if len(mva_options) > 2 else mva_options)
        else:
            rated_mva = random.choice(mva_options[:2] if len(mva_options) > 1 else mva_options)
        
        # Select impedance from vocabulary tokens
        percentz = random.choice(self.param_sets["impedance_values"])
        
        return {
            "rated_MVA": rated_mva,
            "percentZ": percentz
        }
    
    def _select_breaker_parameters(self, voltage: str) -> Dict:
        """Select appropriate breaker parameters for voltage level from vocabulary tokens"""
        # Use vocabulary token sets for all selections
        return {
            "hv_interrupting_kA": random.choice(self.param_sets["available_ka"]),
            "feeder_thermal_A": random.choice(self.param_sets["available_thermal"])
        }
    
    def generate_intent(self, architecture_name: str, complexity_override: str = None) -> Dict:
        """Generate a balanced intent for given architecture"""
        architecture = self.architectures[architecture_name]
        complexity = complexity_override or architecture["complexity"]
        
        # Select voltage combination based on complexity
        kv_in, kv_out = self._select_voltage_combination(complexity)
        
        # Select number of transformers within architecture constraints
        min_tx = architecture["min_transformers"]
        max_tx = architecture["max_transformers"]
        
        if complexity == "low":
            num_transformers = min_tx
        elif complexity == "medium":
            num_transformers = min(max_tx, min_tx + 1)
        else:
            num_transformers = max_tx
        
        # Select transformer parameters
        tx_params = self._select_transformer_parameters(kv_in, kv_out, num_transformers, complexity)
        
        # Select breaker parameters (use highest voltage for ratings)
        all_voltages = kv_in + kv_out
        highest_voltage = max(all_voltages, 
                            key=lambda v: int(v.replace('kv', '').replace('.', '')))
        breaker_params = self._select_breaker_parameters(highest_voltage)
        
        # Select reliability target from architecture options
        reliability_target = random.choice(architecture["reliability"])
        
        return {
            "kv_in": kv_in,
            "kv_out": kv_out,
            "num_transformers": str(num_transformers),
            "rated_MVA": tx_params["rated_MVA"],
            "percentZ": tx_params["percentZ"],
            "hv_interrupting_kA": breaker_params["hv_interrupting_kA"],
            "feeder_thermal_A": breaker_params["feeder_thermal_A"],
            "reliability_target": reliability_target,
            "_architecture": architecture_name,
            "_complexity": complexity
        }
    
    def generate_sequence_for_architecture(self, intent: Dict, architecture_name: str) -> List[str]:
        """Generate action sequence based on architecture pattern"""
        # Reset ID tracking for each sequence to ensure sequential numbering
        self._reset_id_counters()
        
        architecture = self.architectures[architecture_name]
        sequence = ["<START>"]
        
        # Extract intent parameters
        kv_in = intent["kv_in"]
        kv_out = intent["kv_out"]
        num_transformers = int(intent["num_transformers"])
        rated_mva = intent["rated_MVA"]
        percentz = intent["percentZ"]
        
        # Generate components based on architecture
        if architecture_name == "single_transformer_radial":
            sequence.extend(self._generate_single_transformer_radial(intent))
            
        elif architecture_name == "dual_transformer_redundant":
            sequence.extend(self._generate_dual_transformer_redundant(intent))
            
        elif architecture_name == "triple_transformer_high_reliability":
            sequence.extend(self._generate_triple_transformer_high_reliability(intent))
            
        elif architecture_name == "ring_bus":
            sequence.extend(self._generate_ring_bus(intent))
            
        elif architecture_name == "double_busbar":
            sequence.extend(self._generate_double_busbar(intent))
            
        elif architecture_name == "breaker_and_half":
            sequence.extend(self._generate_breaker_and_half(intent))
        
        # Always end with EMIT_SPEC
        sequence.extend(["EMIT_SPEC", "<END>"])
        
        return sequence
    
    def _generate_single_transformer_radial(self, intent: Dict) -> List[str]:
        """Generate simple single transformer radial configuration"""
        sequence = []
        
        kv_in = intent["kv_in"][0]
        kv_out = intent["kv_out"][0]
        rated_mva = intent["rated_MVA"]
        percentz = intent["percentZ"]
        
        # Create buses
        hv_bus_id = self._generate_component_id("bus")
        lv_bus_id = self._generate_component_id("bus")
        
        sequence.extend([
            "ADD_BUS", "id=", hv_bus_id, "kv=", kv_in,
            "ADD_BUS", "id=", lv_bus_id, "kv=", kv_out
        ])
        
        # Add transformer
        tx_id = self._generate_component_id("transformer")
        sequence.extend([
            "ADD_TRANSFORMER", "id=", tx_id, "kind=", "TWO_WINDING",
            "kv_in=", kv_in, "kv_out=", kv_out, "tert_kv=", kv_out,
            "rated_MVA=", rated_mva, "vector_group=", "Dd0", "percentZ=", percentz
        ])
        
        # Simple connection
        sequence.extend([
            "CONNECT", "series=", "[", hv_bus_id, ",", tx_id, ",", lv_bus_id, "]"
        ])
        
        return sequence
    
    def _generate_dual_transformer_redundant(self, intent: Dict) -> List[str]:
        """Generate dual transformer with redundant connections"""
        sequence = []
        
        kv_in = intent["kv_in"][0]
        kv_out = intent["kv_out"]
        rated_mva = intent["rated_MVA"]
        percentz = intent["percentZ"]
        
        # Create buses
        hv_bus_id = self._generate_component_id("bus")
        lv_buses = [self._generate_component_id("bus") for _ in kv_out]
        
        sequence.extend(["ADD_BUS", "id=", hv_bus_id, "kv=", kv_in])
        for i, voltage in enumerate(kv_out):
            sequence.extend(["ADD_BUS", "id=", lv_buses[i], "kv=", voltage])
        
        # Add transformers
        tx_ids = [self._generate_component_id("transformer") for _ in range(2)]
        for i, tx_id in enumerate(tx_ids):
            # Each transformer serves a different LV voltage if available
            lv_voltage = kv_out[i % len(kv_out)]
            sequence.extend([
                "ADD_TRANSFORMER", "id=", tx_id, "kind=", "TWO_WINDING",
                "kv_in=", kv_in, "kv_out=", lv_voltage, "tert_kv=", lv_voltage,
                "rated_MVA=", rated_mva, "vector_group=", "Dd0", "percentZ=", percentz
            ])
        
        # Simple connections without unused breakers
        for i, tx_id in enumerate(tx_ids):
            lv_bus = lv_buses[i % len(lv_buses)]
            sequence.extend([
                "CONNECT", "series=", "[", hv_bus_id, ",", tx_id, ",", lv_bus, "]"
            ])
        
        return sequence
    
    def _generate_triple_transformer_high_reliability(self, intent: Dict) -> List[str]:
        """Generate triple transformer configuration for high reliability"""
        sequence = []
        
        kv_in = intent["kv_in"]
        kv_out = intent["kv_out"]
        rated_mva = intent["rated_MVA"]
        percentz = intent["percentZ"]
        
        # Multiple HV buses for redundancy
        hv_buses = [self._generate_component_id("bus") for _ in kv_in]
        lv_buses = [self._generate_component_id("bus") for _ in kv_out]
        
        # Add HV buses
        for i, voltage in enumerate(kv_in):
            sequence.extend(["ADD_BUS", "id=", hv_buses[i], "kv=", voltage])
        
        # Add LV buses
        for i, voltage in enumerate(kv_out):
            sequence.extend(["ADD_BUS", "id=", lv_buses[i], "kv=", voltage])
        
        # Add three transformers
        tx_ids = [self._generate_component_id("transformer") for _ in range(3)]
        for i, tx_id in enumerate(tx_ids):
            hv_voltage = kv_in[i % len(kv_in)]
            lv_voltage = kv_out[i % len(kv_out)]
            sequence.extend([
                "ADD_TRANSFORMER", "id=", tx_id, "kind=", "TWO_WINDING",
                "kv_in=", hv_voltage, "kv_out=", lv_voltage, "tert_kv=", lv_voltage,
                "rated_MVA=", rated_mva, "vector_group=", "Dd0", "percentZ=", percentz
            ])
        
        # Add breakers for each transformer
        breaker_ids = [self._generate_component_id("breaker") for _ in range(3)]
        for i, breaker_id in enumerate(breaker_ids):
            voltage = kv_in[i % len(kv_in)]
            sequence.extend([
                "ADD_BREAKER", "id=", breaker_id, "kv=", voltage,
                "interrupting_kA=", intent["hv_interrupting_kA"],
                "kind=", "SF6", "continuous_A=", intent["feeder_thermal_A"]
            ])
        
        # Mesh connections for maximum reliability
        for i in range(3):
            hv_bus = hv_buses[i % len(hv_buses)]
            lv_bus = lv_buses[i % len(lv_buses)]
            sequence.extend([
                "CONNECT", "series=", "[", hv_bus, ",", breaker_ids[i], ",", tx_ids[i], ",", lv_bus, "]"
            ])
        
        return sequence
    
    def _generate_ring_bus(self, intent: Dict) -> List[str]:
        """Generate ring bus configuration"""
        sequence = []
        
        kv_in = intent["kv_in"][0]
        kv_out = intent["kv_out"]
        num_transformers = int(intent["num_transformers"])
        rated_mva = intent["rated_MVA"]
        percentz = intent["percentZ"]
        
        # Create ring of buses
        ring_buses = [self._generate_component_id("bus") for _ in range(4)]  # 4-bus ring
        lv_buses = [self._generate_component_id("bus") for _ in kv_out]
        
        # Add ring buses
        for bus_id in ring_buses:
            sequence.extend(["ADD_BUS", "id=", bus_id, "kv=", kv_in])
        
        # Add LV buses
        for i, voltage in enumerate(kv_out):
            sequence.extend(["ADD_BUS", "id=", lv_buses[i], "kv=", voltage])
        
        # Add transformers connected to ring
        tx_ids = [self._generate_component_id("transformer") for _ in range(num_transformers)]
        for i, tx_id in enumerate(tx_ids):
            lv_voltage = kv_out[i % len(kv_out)]
            sequence.extend([
                "ADD_TRANSFORMER", "id=", tx_id, "kind=", "TWO_WINDING",
                "kv_in=", kv_in, "kv_out=", lv_voltage, "tert_kv=", lv_voltage,
                "rated_MVA=", rated_mva, "vector_group=", "Dd0", "percentZ=", percentz
            ])
        
        # Ring connections
        for i in range(len(ring_buses)):
            next_bus = ring_buses[(i + 1) % len(ring_buses)]
            sequence.extend([
                "CONNECT", "series=", "[", ring_buses[i], ",", next_bus, "]"
            ])
        
        # Connect transformers to ring
        for i, tx_id in enumerate(tx_ids):
            ring_bus = ring_buses[i % len(ring_buses)]
            lv_bus = lv_buses[i % len(lv_buses)]
            sequence.extend([
                "CONNECT", "series=", "[", ring_bus, ",", tx_id, ",", lv_bus, "]"
            ])
        
        return sequence
    
    def _generate_double_busbar(self, intent: Dict) -> List[str]:
        """Generate double busbar configuration"""
        sequence = []
        
        kv_in = intent["kv_in"][0]
        kv_out = intent["kv_out"]
        num_transformers = int(intent["num_transformers"])
        rated_mva = intent["rated_MVA"]
        percentz = intent["percentZ"]
        
        # Create double busbars
        bus_a_id = self._generate_component_id("bus")
        bus_b_id = self._generate_component_id("bus")
        lv_buses = [self._generate_component_id("bus") for _ in kv_out]
        
        sequence.extend([
            "ADD_BUS", "id=", bus_a_id, "kv=", kv_in,
            "ADD_BUS", "id=", bus_b_id, "kv=", kv_in
        ])
        
        for i, voltage in enumerate(kv_out):
            sequence.extend(["ADD_BUS", "id=", lv_buses[i], "kv=", voltage])
        
        # Add tie breaker between busbars
        tie_breaker_id = self._generate_component_id("breaker")
        sequence.extend([
            "ADD_BREAKER", "id=", tie_breaker_id, "kv=", kv_in,
            "interrupting_kA=", intent["hv_interrupting_kA"],
            "kind=", "SF6", "continuous_A=", intent["feeder_thermal_A"]
        ])
        
        # Connect tie breaker
        sequence.extend([
            "CONNECT", "series=", "[", bus_a_id, ",", tie_breaker_id, ",", bus_b_id, "]"
        ])
        
        # Add transformers alternating between buses
        tx_ids = [self._generate_component_id("transformer") for _ in range(num_transformers)]
        for i, tx_id in enumerate(tx_ids):
            lv_voltage = kv_out[i % len(kv_out)]
            hv_bus = bus_a_id if i % 2 == 0 else bus_b_id
            
            sequence.extend([
                "ADD_TRANSFORMER", "id=", tx_id, "kind=", "TWO_WINDING",
                "kv_in=", kv_in, "kv_out=", lv_voltage, "tert_kv=", lv_voltage,
                "rated_MVA=", rated_mva, "vector_group=", "Dd0", "percentZ=", percentz
            ])
            
            # Connect transformer to alternating busbar
            lv_bus = lv_buses[i % len(lv_buses)]
            sequence.extend([
                "CONNECT", "series=", "[", hv_bus, ",", tx_id, ",", lv_bus, "]"
            ])
        
        return sequence
    
    def _generate_breaker_and_half(self, intent: Dict) -> List[str]:
        """Generate breaker and a half scheme"""
        sequence = []
        
        kv_in = intent["kv_in"][0]
        kv_out = intent["kv_out"]
        num_transformers = min(int(intent["num_transformers"]), 4)  # Limit for this scheme
        rated_mva = intent["rated_MVA"]
        percentz = intent["percentZ"]
        
        # Create main buses
        main_bus_1 = self._generate_component_id("bus")
        main_bus_2 = self._generate_component_id("bus")
        lv_buses = [self._generate_component_id("bus") for _ in kv_out]
        
        sequence.extend([
            "ADD_BUS", "id=", main_bus_1, "kv=", kv_in,
            "ADD_BUS", "id=", main_bus_2, "kv=", kv_in
        ])
        
        for i, voltage in enumerate(kv_out):
            sequence.extend(["ADD_BUS", "id=", lv_buses[i], "kv=", voltage])
        
        # Add breakers (1.5 breakers per circuit)
        breaker_ids = [self._generate_component_id("breaker") for _ in range(num_transformers + 1)]
        for breaker_id in breaker_ids:
            sequence.extend([
                "ADD_BREAKER", "id=", breaker_id, "kv=", kv_in,
                "interrupting_kA=", intent["hv_interrupting_kA"],
                "kind=", "SF6", "continuous_A=", intent["feeder_thermal_A"]
            ])
        
        # Add transformers
        tx_ids = [self._generate_component_id("transformer") for _ in range(num_transformers)]
        for i, tx_id in enumerate(tx_ids):
            lv_voltage = kv_out[i % len(kv_out)]
            sequence.extend([
                "ADD_TRANSFORMER", "id=", tx_id, "kind=", "TWO_WINDING",
                "kv_in=", kv_in, "kv_out=", lv_voltage, "tert_kv=", lv_voltage,
                "rated_MVA=", rated_mva, "vector_group=", "Dd0", "percentZ=", percentz
            ])
        
        # Breaker and half connections (each circuit has 1.5 breakers)
        for i in range(num_transformers):
            # Connect transformer between two breakers
            breaker_1 = breaker_ids[i]
            breaker_2 = breaker_ids[i + 1]
            tx_id = tx_ids[i]
            lv_bus = lv_buses[i % len(lv_buses)]
            
            if i % 2 == 0:
                # Connect to main_bus_1
                sequence.extend([
                    "CONNECT", "series=", "[", main_bus_1, ",", breaker_1, ",", tx_id, ",", breaker_2, ",", lv_bus, "]"
                ])
            else:
                # Connect to main_bus_2
                sequence.extend([
                    "CONNECT", "series=", "[", main_bus_2, ",", breaker_1, ",", tx_id, ",", breaker_2, ",", lv_bus, "]"
                ])
        
        return sequence
    
    def generate_balanced_dataset(self, total_samples: int, output_file: str = None) -> List[Dict]:
        """Generate a balanced dataset with varied architectures and complexity levels"""
        if output_file:
            self.output_file = output_file
        
        print(f"Generating {total_samples} balanced samples...")
        
        # Define balance targets
        complexity_distribution = {
            "low": 0.4,      # 40% simple cases
            "medium": 0.35,  # 35% medium complexity
            "high": 0.25     # 25% high complexity
        }
        
        architecture_names = list(self.architectures.keys())
        samples_per_complexity = {
            complexity: int(total_samples * ratio)
            for complexity, ratio in complexity_distribution.items()
        }
        
        # Adjust for rounding errors
        total_assigned = sum(samples_per_complexity.values())
        if total_assigned < total_samples:
            samples_per_complexity["medium"] += total_samples - total_assigned
        
        dataset = []
        generation_stats = defaultdict(int)
        
        # Generate samples for each complexity level
        for complexity, num_samples in samples_per_complexity.items():
            print(f"Generating {num_samples} {complexity} complexity samples...")
            
            samples_per_arch = num_samples // len(architecture_names)
            remainder = num_samples % len(architecture_names)
            
            for i, arch_name in enumerate(architecture_names):
                arch_samples = samples_per_arch + (1 if i < remainder else 0)
                
                for _ in range(arch_samples):
                    try:
                        # Generate intent
                        intent = self.generate_intent(arch_name, complexity)
                        
                        # Generate sequence
                        sequence = self.generate_sequence_for_architecture(intent, arch_name)
                        
                        # Validate with grammar enforcer
                        is_valid, error_msg = self._validate_sequence(sequence)
                        if not is_valid:
                            print(f"Warning: Invalid sequence generated for {arch_name}: {error_msg}")
                            # Try to repair or skip
                            continue
                        
                        # Remove metadata from intent before saving
                        clean_intent = {k: v for k, v in intent.items() 
                                      if not k.startswith('_')}
                        
                        sample = {
                            "intent": clean_intent,
                            "sequence": sequence,  # Keep <START> and <END> tokens
                            "_metadata": {
                                "architecture": arch_name,
                                "complexity": complexity,
                                "generation_method": "synthetic"
                            }
                        }
                        
                        dataset.append(sample)
                        generation_stats[f"{complexity}_{arch_name}"] += 1
                        
                    except Exception as e:
                        print(f"Error generating sample for {arch_name} ({complexity}): {e}")
                        continue
        
        print(f"Successfully generated {len(dataset)} samples")
        print("\nGeneration Statistics:")
        for key, count in generation_stats.items():
            print(f"  {key}: {count}")
        
        # Balance check
        self._print_balance_analysis(dataset)
        
        # Save dataset
        if self.output_file:
            self.save_dataset(dataset)
        
        return dataset
    
    def _validate_sequence(self, sequence: List[str]) -> Tuple[bool, str]:
        """Validate sequence using grammar enforcer"""
        try:
            # Use grammar enforcer to validate
            token_ids, is_valid, error_message = self.grammar.process_sequence_for_training(
                sequence, repair_if_invalid=False
            )
            return is_valid, error_message
        except Exception as e:
            return False, str(e)
    
    def _print_balance_analysis(self, dataset: List[Dict]):
        """Print analysis of dataset balance"""
        print("\n" + "="*50)
        print("DATASET BALANCE ANALYSIS")
        print("="*50)
        
        # Architecture balance
        arch_counts = defaultdict(int)
        complexity_counts = defaultdict(int)
        voltage_combinations = defaultdict(int)
        mva_ratings = defaultdict(int)
        
        for sample in dataset:
            metadata = sample.get("_metadata", {})
            intent = sample["intent"]
            
            arch_counts[metadata.get("architecture", "unknown")] += 1
            complexity_counts[metadata.get("complexity", "unknown")] += 1
            
            # Voltage combinations
            kv_combo = f"{'+'.join(intent['kv_in'])} → {'+'.join(intent['kv_out'])}"
            voltage_combinations[kv_combo] += 1
            
            mva_ratings[intent["rated_MVA"]] += 1
        
        print(f"\nArchitecture Distribution ({len(dataset)} total):")
        for arch, count in sorted(arch_counts.items()):
            percentage = (count / len(dataset)) * 100
            print(f"  {arch:<30}: {count:>4} ({percentage:>5.1f}%)")
        
        print("\nComplexity Distribution:")
        for complexity, count in sorted(complexity_counts.items()):
            percentage = (count / len(dataset)) * 100
            print(f"  {complexity:<10}: {count:>4} ({percentage:>5.1f}%)")
        
        print("\nTop Voltage Combinations:")
        for combo, count in sorted(voltage_combinations.items(), 
                                 key=lambda x: x[1], reverse=True)[:10]:
            percentage = (count / len(dataset)) * 100
            print(f"  {combo:<25}: {count:>4} ({percentage:>5.1f}%)")
        
        print("\nMVA Rating Distribution:")
        for mva, count in sorted(mva_ratings.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(dataset)) * 100
            print(f"  {mva:<10}: {count:>4} ({percentage:>5.1f}%)")
    
    def save_dataset(self, dataset: List[Dict]):
        """Save dataset to JSON file"""
        # Remove metadata before saving
        clean_dataset = []
        for sample in dataset:
            clean_sample = {
                "intent": sample["intent"],
                "sequence": sample["sequence"]
            }
            clean_dataset.append(clean_sample)
        
        with open(self.output_file, 'w') as f:
            json.dump(clean_dataset, f, indent=2)
        
        print(f"\nDataset saved to: {self.output_file}")
        
    def generate_test_cases(self) -> List[Dict]:
        """Generate specific test cases for model validation"""
        test_cases = []
        
        # Test case 1: Minimal single transformer
        test_cases.append({
            "name": "minimal_single_transformer",
            "intent": {
                "kv_in": ["69kv"],
                "kv_out": ["13.8kv"],
                "num_transformers": "1",
                "rated_MVA": "25mva",
                "percentZ": "6percentZ",
                "hv_interrupting_kA": "31.5kA",
                "feeder_thermal_A": "600A",
                "reliability_target": "N_0"
            }
        })
        
        # Test case 2: High complexity multi-level
        test_cases.append({
            "name": "complex_multi_level",
            "intent": {
                "kv_in": ["345kv", "230kv"],
                "kv_out": ["138kv", "69kv", "25kv", "13.8kv"],
                "num_transformers": "3",
                "rated_MVA": "300mva",
                "percentZ": "10percentZ",
                "hv_interrupting_kA": "63kA",
                "feeder_thermal_A": "1200A",
                "reliability_target": "N_2"
            }
        })
        
        # Generate sequences for test cases
        for test_case in test_cases:
            # Find appropriate architecture
            intent = test_case["intent"]
            num_tx = int(intent["num_transformers"])
            reliability = intent["reliability_target"]
            
            # Select architecture based on requirements
            if num_tx == 1:
                arch_name = "single_transformer_radial"
            elif reliability in ["N_2", "REDUNDANT"]:
                arch_name = "triple_transformer_high_reliability"
            else:
                arch_name = "dual_transformer_redundant"
            
            sequence = self.generate_sequence_for_architecture(intent, arch_name)
            test_case["sequence"] = sequence  # Keep <START> and <END> tokens
            test_case["expected_architecture"] = arch_name
        
        return test_cases


def main():
    """Main generation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic substation corpus')
    parser.add_argument('--samples', type=int, default=5000, 
                       help='Number of samples to generate')
    parser.add_argument('--output', type=str, default='../../data/synthetic_corpus.json',
                       help='Output file path')
    parser.add_argument('--test-cases', action='store_true',
                       help='Generate test cases instead of full corpus')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)
    
    generator = DataGenerator(args.output)
    
    if args.test_cases:
        print("Generating test cases...")
        test_cases = generator.generate_test_cases()
        
        test_output = args.output.replace('.json', '_test_cases.json')
        with open(test_output, 'w') as f:
            json.dump(test_cases, f, indent=2)
        print(f"Test cases saved to: {test_output}")
        
        # Print test cases
        for case in test_cases:
            print(f"\n{case['name']}:")
            print(f"  Architecture: {case['expected_architecture']}")
            print(f"  Intent: {case['intent']}")
            print(f"  Sequence length: {len(case['sequence'])}")
    
    else:
        # Generate full corpus
        dataset = generator.generate_balanced_dataset(args.samples)
        print(f"\nGenerated {len(dataset)} samples")
        print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()



