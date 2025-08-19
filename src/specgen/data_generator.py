"""
data_generator.py

Advanced synthetic data generation for electrical substation training.
Creates realistic, diverse training examples based on engineering patterns.

Author: Gordon Doore
Date Created: 2025-08-19
"""

import random
import json
from typing import Dict, List
from dataclasses import dataclass, asdict
from enum import Enum

class SubstationType(Enum):
    TRANSMISSION = "transmission"  # 230kV+
    SUB_TRANSMISSION = "sub_transmission"  # 69-138kV  
    DISTRIBUTION = "distribution"  # <69kV
    INDUSTRIAL = "industrial"  # Special purpose
    RENEWABLE = "renewable"  # Solar/wind integration

class ReliabilityLevel(Enum):
    BASIC = "basic"
    STANDARD = "standard" 
    HIGH = "high"
    CRITICAL = "critical"  # Hospitals, data centers

@dataclass
class SubstationSpec:
    """Complete specification for a substation"""
    # Basic electrical parameters
    hv_voltage: float
    lv_voltage: float
    transformer_mva: float
    hv_scheme: str  # "single_bus", "double_bus", "ring", "breaker_half"
    lv_scheme: str
    substation_type: SubstationType
    reliability: ReliabilityLevel
    num_feeders: int
    feeder_types: List[str]  # ["residential", "commercial", "industrial"]
    location: str  # "urban", "rural", "industrial"
    standards: List[str]  # ["IEEE", "IEC", "utility_specific"]
    budget_level: str  # "minimal", "standard", "premium"
    
    # Optional parameters with defaults
    num_transformers: int = 1
    lifecycle_years: int = 30

class ElectricalDataGenerator:
    """Generate realistic electrical substation training data"""
    
    def __init__(self):
        self.voltage_pairs = self._init_voltage_pairs()
        self.equipment_standards = self._init_equipment_standards()
        
    def _init_voltage_pairs(self) -> Dict:
        """Standard voltage transformation pairs"""
        return {
            SubstationType.TRANSMISSION: [
                (500, 230), (345, 138), (230, 138), (345, 115)
            ],
            SubstationType.SUB_TRANSMISSION: [
                (138, 34.5), (138, 25), (115, 34.5), (115, 25), 
                (69, 13.8), (69, 25), (138, 13.8)
            ],
            SubstationType.DISTRIBUTION: [
                (34.5, 13.8), (25, 13.8), (13.8, 4.16), (34.5, 4.16)
            ],
            SubstationType.INDUSTRIAL: [
                (138, 13.8), (69, 13.8), (25, 4.16), (138, 4.16)
            ],
            SubstationType.RENEWABLE: [
                (345, 34.5), (138, 34.5), (69, 34.5)  # Generation step-up
            ]
        }
    
    def _init_equipment_standards(self) -> Dict:
        """Equipment selection based on voltage and application"""
        return {
            "protection": {
                "basic": ["breaker", "fuse", "recloser"],
                "standard": ["breaker", "relay", "ct", "vt"], 
                "high": ["breaker", "relay", "ct", "vt", "surge_arrester"],
                "critical": ["breaker", "relay", "ct", "vt", "surge_arrester", "backup_protection"]
            },
            "bus_schemes": {
                SubstationType.TRANSMISSION: ["ring", "breaker_half", "double_bus"],
                SubstationType.SUB_TRANSMISSION: ["single_bus", "double_bus", "main_transfer"],
                SubstationType.DISTRIBUTION: ["single_bus", "radial"],
                SubstationType.INDUSTRIAL: ["single_bus", "double_bus"],
                SubstationType.RENEWABLE: ["single_bus", "double_bus"]
            }
        }
    
    def generate_realistic_spec(self) -> SubstationSpec:
        """Generate a realistic substation specification"""
        
        # Choose substation type based on realistic distribution
        type_weights = {
            SubstationType.DISTRIBUTION: 0.50,  # Most common
            SubstationType.SUB_TRANSMISSION: 0.30,
            SubstationType.TRANSMISSION: 0.10,
            SubstationType.INDUSTRIAL: 0.08,
            SubstationType.RENEWABLE: 0.02  # Growing but still rare
        }
        
        substation_type = random.choices(
            list(type_weights.keys()),
            weights=list(type_weights.values())
        )[0]
        
        # Choose voltage pair appropriate for type
        hv, lv = random.choice(self.voltage_pairs[substation_type])
        
        # Size transformer based on voltage and type
        mva_options = self._get_mva_options(substation_type, hv)
        transformer_mva = random.choice(mva_options)
        
        # Reliability based on application
        reliability_weights = self._get_reliability_weights(substation_type)
        reliability = random.choices(
            list(reliability_weights.keys()),
            weights=list(reliability_weights.values())
        )[0]
        
        # Bus scheme based on reliability and type
        hv_schemes = self.equipment_standards["bus_schemes"][substation_type]
        hv_scheme = random.choice(hv_schemes)
        
        # LV scheme is usually simpler
        lv_scheme = "radial" if substation_type in [SubstationType.DISTRIBUTION] else "single_bus"
        
        # Feeder count based on size and type
        num_feeders = self._calc_feeder_count(substation_type, transformer_mva)
        feeder_types = self._generate_feeder_types(substation_type, num_feeders)
        
        # Other realistic parameters
        location = random.choices(
            ["urban", "rural", "industrial"], 
            weights=[0.4, 0.4, 0.2]
        )[0]
        
        budget_weights = {"minimal": 0.3, "standard": 0.5, "premium": 0.2}
        budget_level = random.choices(
            list(budget_weights.keys()),
            weights=list(budget_weights.values())
        )[0]
        
        return SubstationSpec(
            hv_voltage=hv,
            lv_voltage=lv,
            transformer_mva=transformer_mva,
            hv_scheme=hv_scheme,
            lv_scheme=lv_scheme,
            substation_type=substation_type,
            reliability=reliability,
            num_feeders=num_feeders,
            feeder_types=feeder_types,
            location=location,
            standards=["IEEE"],  # Simplified for now
            budget_level=budget_level
        )
    
    def _get_mva_options(self, substation_type: SubstationType, voltage: float) -> List[float]:
        """Get realistic MVA options for given type and voltage"""
        if substation_type == SubstationType.TRANSMISSION:
            return [300, 500, 750, 1000]
        elif substation_type == SubstationType.SUB_TRANSMISSION:
            if voltage >= 138:
                return [50, 75, 100, 150, 200]
            else:
                return [25, 50, 75, 100]
        elif substation_type == SubstationType.DISTRIBUTION:
            return [10, 15, 25, 37.5, 50]
        elif substation_type == SubstationType.INDUSTRIAL:
            return [25, 50, 75, 100, 150]
        else:  # RENEWABLE
            return [50, 100, 150, 200, 300]
    
    def _get_reliability_weights(self, substation_type: SubstationType) -> Dict:
        """Reliability requirements by substation type"""
        if substation_type == SubstationType.TRANSMISSION:
            return {ReliabilityLevel.HIGH: 0.6, ReliabilityLevel.CRITICAL: 0.4}
        elif substation_type == SubstationType.SUB_TRANSMISSION:
            return {ReliabilityLevel.STANDARD: 0.5, ReliabilityLevel.HIGH: 0.5}
        elif substation_type == SubstationType.DISTRIBUTION:
            return {ReliabilityLevel.BASIC: 0.3, ReliabilityLevel.STANDARD: 0.7}
        elif substation_type == SubstationType.INDUSTRIAL:
            return {ReliabilityLevel.STANDARD: 0.4, ReliabilityLevel.HIGH: 0.4, ReliabilityLevel.CRITICAL: 0.2}
        else:  # RENEWABLE
            return {ReliabilityLevel.STANDARD: 0.6, ReliabilityLevel.HIGH: 0.4}
    
    def _calc_feeder_count(self, substation_type: SubstationType, mva: float) -> int:
        """Calculate realistic feeder count"""
        base_feeders = {
            SubstationType.TRANSMISSION: 1,  # Usually just transmission lines
            SubstationType.SUB_TRANSMISSION: max(2, int(mva / 25)),  # ~25 MVA per feeder
            SubstationType.DISTRIBUTION: max(4, int(mva / 8)),   # ~8 MVA per feeder
            SubstationType.INDUSTRIAL: max(2, int(mva / 15)),    # ~15 MVA per feeder  
            SubstationType.RENEWABLE: max(1, int(mva / 50))      # Collector feeders
        }
        
        base = base_feeders[substation_type]
        # Add some randomness
        return max(1, base + random.randint(-1, 2))
    
    def _generate_feeder_types(self, substation_type: SubstationType, count: int) -> List[str]:
        """Generate realistic mix of feeder types"""
        if substation_type == SubstationType.DISTRIBUTION:
            types = ["residential", "commercial", "industrial"]
            weights = [0.5, 0.3, 0.2]
        elif substation_type == SubstationType.INDUSTRIAL:
            types = ["industrial", "commercial"]
            weights = [0.8, 0.2]
        else:
            types = ["transmission_line", "feeder"]
            weights = [0.7, 0.3]
        
        return [random.choices(types, weights=weights)[0] for _ in range(count)]
    
    def spec_to_intent(self, spec: SubstationSpec) -> Dict:
        """Convert SubstationSpec to model intent format"""
        return {
            "from_kv": spec.hv_voltage,
            "to_kv": spec.lv_voltage,
            "rating_mva": spec.transformer_mva,
            "scheme_hv": spec.hv_scheme,
            "scheme_lv": spec.lv_scheme,
            "lv_feeders": spec.num_feeders,
            "protection": spec.reliability.value,
            "substation_type": spec.substation_type.value,
            "location": spec.location,
            "budget": spec.budget_level
        }
    
    def generate_training_dataset(self, num_examples: int = 1000) -> List[Dict]:
        """Generate complete training dataset"""
        dataset = []
        
        print(f"Generating {num_examples} realistic substation specifications...")
        
        for i in range(num_examples):
            if i % 100 == 0:
                print(f"Generated {i}/{num_examples}")
            
            # Generate realistic spec
            spec = self.generate_realistic_spec()
            intent = self.spec_to_intent(spec)
            
            # Add to dataset with metadata
            dataset.append({
                "intent": intent,
                "spec": asdict(spec),
                "id": f"substation_{i:04d}"
            })
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], filename: str):
        """Save dataset to JSON file"""
        with open(filename, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)
        print(f"Saved {len(dataset)} examples to {filename}")
    
    def analyze_dataset(self, dataset: List[Dict]):
        """Print dataset statistics"""
        print(f"\n=== Dataset Analysis ({len(dataset)} examples) ===")
        
        # Type distribution
        types = [ex["spec"]["substation_type"] for ex in dataset]
        type_counts = {t: types.count(t) for t in set(types)}
        print("Substation Types:", type_counts)
        
        # Voltage distribution  
        voltages = [(ex["intent"]["from_kv"], ex["intent"]["to_kv"]) for ex in dataset]
        voltage_counts = {v: voltages.count(v) for v in set(voltages)}
        print("Top Voltage Pairs:", sorted(voltage_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # MVA distribution
        mvas = [ex["intent"]["rating_mva"] for ex in dataset]
        print(f"MVA Range: {min(mvas)} - {max(mvas)} (avg: {sum(mvas)/len(mvas):.1f})")


def demo_data_generation():
    """Demonstrate realistic data generation"""
    
    generator = ElectricalDataGenerator()
    
    # Generate some examples
    print("=== Sample Generated Specifications ===")
    for i in range(5):
        spec = generator.generate_realistic_spec()
        intent = generator.spec_to_intent(spec)
        
        print(f"\nExample {i+1}:")
        print(f"  Type: {spec.substation_type.value}")
        print(f"  Voltage: {spec.hv_voltage}kV -> {spec.lv_voltage}kV")
        print(f"  Power: {spec.transformer_mva} MVA")
        print(f"  Reliability: {spec.reliability.value}")
        print(f"  Feeders: {spec.num_feeders} ({', '.join(spec.feeder_types)})")
        print(f"  Intent: {json.dumps(intent, indent=4)}")
    
    # Generate full dataset
    dataset = generator.generate_training_dataset(1000)
    generator.analyze_dataset(dataset)
    
    # Save to file
    generator.save_dataset(dataset, "realistic_substation_data.json")
    
    return dataset


if __name__ == "__main__":
    demo_data_generation()
