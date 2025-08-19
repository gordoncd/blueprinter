"""
Simple vocabulary-compliant corpus generator.
Generates training data that strictly adheres to the model's vocabulary.
"""

import json
import random
from typing import Dict, List

# EXACT vocabulary from the model - must match exactly
VOCAB = [
    # Special tokens
    "<START>", "<END>", "<PAD>", "<UNK>",
    
    # Core actions
    "ADD_BUS", "ADD_TRANSFORMER", "ADD_BAY", "APPEND_STEP", 
    "ADD_CONNECTION", "EMIT_SPEC",
    
    # Common parameters
    "345KV", "220KV", "138KV", "115KV", "69KV", "66KV", "46KV", "34.5KV", "25KV", "13.8KV",
    "100MVA", "50MVA", "75MVA", "150MVA",
    "HV", "LV", "MAIN", "TRANSFORMER_BAY", "FEEDER_BAY",
    "BREAKER", "BUS_ISOLATOR", "LINE_ISOLATOR", "CT",
    "TWO_WINDING", "DOUBLE_BUSBAR", "SINGLE_BUSBAR",
    
    # Connection indicators
    "CONNECT_BUS", "CONNECT_TX_HV", "CONNECT_TX_LV",
    
    # Semantic ID system - grouped by component type
    # Buses: 101-120 (20 buses)
    "BUS_101", "BUS_102", "BUS_103", "BUS_104", "BUS_105", "BUS_106", "BUS_107", "BUS_108", "BUS_109", "BUS_110",
    "BUS_111", "BUS_112", "BUS_113", "BUS_114", "BUS_115", "BUS_116", "BUS_117", "BUS_118", "BUS_119", "BUS_120",
    
    # Transformers: 201-210 (10 transformers)
    "TX_201", "TX_202", "TX_203", "TX_204", "TX_205", "TX_206", "TX_207", "TX_208", "TX_209", "TX_210",
    
    # Bays: 301-340 (40 bays total - mix of HV and LV)
    "BAY_301", "BAY_302", "BAY_303", "BAY_304", "BAY_305", "BAY_306", "BAY_307", "BAY_308", "BAY_309", "BAY_310",
    "BAY_311", "BAY_312", "BAY_313", "BAY_314", "BAY_315", "BAY_316", "BAY_317", "BAY_318", "BAY_319", "BAY_320",
    "BAY_321", "BAY_322", "BAY_323", "BAY_324", "BAY_325", "BAY_326", "BAY_327", "BAY_328", "BAY_329", "BAY_330",
    "BAY_331", "BAY_332", "BAY_333", "BAY_334", "BAY_335", "BAY_336", "BAY_337", "BAY_338", "BAY_339", "BAY_340"
]

VOCAB_SET = set(VOCAB)

# Realistic voltage pairs that make engineering sense
VOLTAGE_PAIRS = [
    ("345KV", "138KV"), ("345KV", "115KV"),
    ("220KV", "138KV"), ("220KV", "69KV"), ("220KV", "34.5KV"),
    ("138KV", "69KV"), ("138KV", "34.5KV"), ("138KV", "25KV"), ("138KV", "13.8KV"),
    ("115KV", "34.5KV"), ("115KV", "25KV"), ("115KV", "13.8KV"),
    ("69KV", "13.8KV"), ("69KV", "25KV"),
    ("46KV", "13.8KV"),
    ("34.5KV", "13.8KV"),
    ("25KV", "13.8KV")
]

# Realistic MVA ratings for different voltage levels
VOLTAGE_MVA_MAP = {
    "345KV": ["150MVA", "100MVA"],
    "220KV": ["150MVA", "100MVA", "75MVA"],
    "138KV": ["100MVA", "75MVA", "50MVA"],
    "115KV": ["75MVA", "50MVA"],
    "69KV": ["75MVA", "50MVA"],
    "66KV": ["50MVA"],
    "46KV": ["50MVA"],
    "34.5KV": ["50MVA", "75MVA"],
    "25KV": ["50MVA"],
    "13.8KV": ["50MVA"]
}

def validate_token(token: str) -> bool:
    """Validate that a token is in the vocabulary"""
    return token in VOCAB_SET

def create_diverse_intent() -> Dict:
    """Create a diverse intent using only vocabulary-compatible values"""
    
    # Pick realistic voltage pair
    hv_voltage, lv_voltage = random.choice(VOLTAGE_PAIRS)
    
    # Pick appropriate MVA rating for this HV voltage
    possible_mva = VOLTAGE_MVA_MAP.get(hv_voltage, ["50MVA"])
    mva_rating = random.choice(possible_mva)
    
    # Choose scheme based on voltage level (higher voltage more likely double busbar)
    hv_voltage_level = int(hv_voltage.replace("KV", "").replace(".", ""))
    if hv_voltage_level >= 138:
        scheme_hv = random.choice(["double_busbar", "single_busbar"])
    else:
        scheme_hv = random.choice(["single_busbar", "single_busbar", "double_busbar"])
    
    # Choose number of feeders (2-8, biased toward lower numbers)
    lv_feeders = random.choices([2, 3, 4, 5, 6, 7, 8], weights=[25, 20, 15, 15, 10, 10, 5])[0]
    
    # Choose protection level
    protection = random.choice(["basic", "standard", "enhanced"])
    
    # Convert voltage strings to numeric values for intent features
    hv_numeric = float(hv_voltage.replace("KV", ""))  # "138KV" -> 138.0
    lv_numeric = float(lv_voltage.replace("KV", ""))  # "13.8KV" -> 13.8
    mva_numeric = int(mva_rating.replace("MVA", ""))  # "100MVA" -> 100
    
    return {
        # Intent values are numeric (for model training)
        "from_kv": hv_numeric,     # e.g., 138.0
        "to_kv": lv_numeric,       # e.g., 13.8
        "rating_mva": mva_numeric, # e.g., 100
        "scheme_hv": scheme_hv,    # e.g., "double_busbar"
        "lv_feeders": lv_feeders,
        "protection": protection,
        # Store vocab tokens for action sequence generation
        "_vocab_hv": hv_voltage,   # e.g., "138KV"
        "_vocab_lv": lv_voltage,   # e.g., "13.8KV"
        "_vocab_mva": mva_rating   # e.g., "100MVA"
    }

def generate_action_sequence(intent: Dict) -> List[str]:
    """Generate action sequence that uses ONLY vocabulary tokens"""
    
    tokens = ["<START>"]
    
    # Get vocab-compliant values from intent
    hv_voltage = intent["_vocab_hv"]    # e.g., "138KV"
    lv_voltage = intent["_vocab_lv"]    # e.g., "13.8KV" 
    transformer_mva = intent["_vocab_mva"]  # e.g., "100MVA"
    scheme_hv = intent["scheme_hv"]     # e.g., "double_busbar"
    lv_feeders = intent["lv_feeders"]   # e.g., 4
    protection = intent["protection"]   # e.g., "standard"
    
    # Map scheme to vocabulary token
    if scheme_hv == "double_busbar":
        scheme_token = "DOUBLE_BUSBAR"
    else:
        scheme_token = "SINGLE_BUSBAR"
    
    # Validate all tokens before using them
    assert validate_token(hv_voltage), f"Invalid HV voltage: {hv_voltage}"
    assert validate_token(lv_voltage), f"Invalid LV voltage: {lv_voltage}"
    assert validate_token(transformer_mva), f"Invalid MVA: {transformer_mva}"
    assert validate_token(scheme_token), f"Invalid scheme: {scheme_token}"
    
    # Add HV buses based on scheme
    if scheme_hv == "double_busbar":
        tokens.extend([
            "ADD_BUS", hv_voltage, "HV", "MAIN", "BUS_101",
            "ADD_BUS", hv_voltage, "HV", "MAIN", "BUS_102"
        ])
        main_hv_bus = "BUS_101"
    else:  # SINGLE_BUSBAR
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
        "ADD_TRANSFORMER", "TWO_WINDING", hv_voltage, lv_voltage, transformer_mva, "TX_201"
    ])
    
    # Add transformer bay
    tokens.extend([
        "ADD_BAY", "HV", hv_voltage, "TRANSFORMER_BAY", "BAY_301"
    ])
    
    # Add protection equipment based on protection level
    if protection == "enhanced":
        tokens.extend([
            "APPEND_STEP", "BAY_301", "BUS_ISOLATOR",
            "APPEND_STEP", "BAY_301", "CT",
            "APPEND_STEP", "BAY_301", "BREAKER",
            "APPEND_STEP", "BAY_301", "LINE_ISOLATOR"
        ])
    elif protection == "basic":
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
    
    # Add feeder bays (limit to available bay IDs)
    max_feeders = min(lv_feeders, 8)  # Cap at 8 feeders
    for i in range(max_feeders):
        bay_id = f"BAY_{302 + i}"
        
        tokens.extend([
            "ADD_BAY", "LV", lv_voltage, "FEEDER_BAY", bay_id
        ])
        
        # Vary protection based on feeder number and protection level
        if protection == "enhanced":
            tokens.extend([
                "APPEND_STEP", bay_id, "BUS_ISOLATOR",
                "APPEND_STEP", bay_id, "BREAKER",
                "APPEND_STEP", bay_id, "CT"
            ])
        elif protection == "basic":
            tokens.extend([
                "APPEND_STEP", bay_id, "BREAKER",
                "APPEND_STEP", bay_id, "BUS_ISOLATOR"
            ])
        else:  # standard protection
            if i == 0:  # First feeder gets more protection
                tokens.extend([
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR",
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "CT"
                ])
            else:
                tokens.extend([
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR",
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "LINE_ISOLATOR"
                ])
    
    # Add connections
    tokens.extend([
        "CONNECT_BUS", main_hv_bus, "BAY_301",
        "CONNECT_TX_HV", "TX_201", main_hv_bus,
        "CONNECT_TX_LV", "TX_201", "BUS_103"
    ])
    
    # Add spec emission
    tokens.extend(["EMIT_SPEC", "<END>"])
    
    # CRITICAL: Validate every single token before returning
    invalid_tokens = [token for token in tokens if not validate_token(token)]
    if invalid_tokens:
        raise ValueError(f"Generated invalid tokens: {invalid_tokens}")
    
    return tokens

def generate_corpus(num_examples: int = 1000) -> List[Dict]:
    """Generate corpus with perfect vocabulary compliance"""
    
    print(f"Generating {num_examples} vocabulary-compliant training examples...")
    
    corpus = []
    
    for i in range(num_examples):
        if i % 100 == 0:
            print(f"Generated {i}/{num_examples} examples")
        
        try:
            # Create diverse intent
            intent = create_diverse_intent()
            
            # Generate action sequence
            action_sequence = generate_action_sequence(intent)
            
            # Create corpus entry
            corpus_entry = {
                "id": f"substation_{i:04d}",
                "intent": intent,
                "action_sequence": action_sequence,
                "sequence_length": len(action_sequence),
                "vocabulary_compliant": True
            }
            
            corpus.append(corpus_entry)
            
        except Exception as e:
            print(f"Error generating example {i}: {e}")
            continue
    
    print(f"Successfully generated {len(corpus)} vocabulary-compliant examples")
    return corpus

def analyze_corpus(corpus: List[Dict]):
    """Analyze corpus for vocabulary compliance"""
    
    print("\n=== Vocabulary Compliance Analysis ===")
    
    all_tokens = []
    invalid_tokens = []
    
    for entry in corpus:
        tokens = entry["action_sequence"]
        all_tokens.extend(tokens)
        
        # Check each token
        for token in tokens:
            if not validate_token(token):
                invalid_tokens.append(token)
    
    # Calculate stats
    total_tokens = len(all_tokens)
    unique_tokens = len(set(all_tokens))
    
    print(f"Total tokens generated: {total_tokens}")
    print(f"Unique tokens generated: {unique_tokens}")
    print(f"Vocabulary size: {len(VOCAB)}")
    print(f"Invalid tokens found: {len(invalid_tokens)} ({len(invalid_tokens)/total_tokens*100:.2f}%)")
    
    if invalid_tokens:
        from collections import Counter
        invalid_counts = Counter(invalid_tokens)
        print("Most common invalid tokens:")
        for token, count in invalid_counts.most_common(5):
            print(f"  {token}: {count} times")
    else:
        print("✅ Perfect vocabulary compliance!")
    
    # Check voltage/MVA distribution
    available_voltages = ["345KV", "220KV", "138KV", "115KV", "69KV", "66KV", "46KV", "34.5KV", "25KV", "13.8KV"]
    available_mva = ["100MVA", "50MVA", "75MVA", "150MVA"]
    
    voltages_used = [token for token in all_tokens if token in available_voltages]
    mva_used = [token for token in all_tokens if token in available_mva]
    
    from collections import Counter
    voltage_dist = Counter(voltages_used)
    mva_dist = Counter(mva_used)
    
    print("\nVoltage distribution:")
    for voltage, count in voltage_dist.most_common():
        print(f"  {voltage}: {count} ({count/len(voltages_used)*100:.1f}%)")
    
    print("\nMVA distribution:")
    for mva, count in mva_dist.most_common():
        print(f"  {mva}: {count} ({count/len(mva_used)*100:.1f}%)")

def sample_corpus(corpus: List[Dict], num_samples: int = 3):
    """Show sample entries from corpus"""
    print("\n=== Sample Corpus Entries ===")
    
    samples = random.sample(corpus, min(num_samples, len(corpus)))
    
    for i, entry in enumerate(samples):
        print(f"\nSample {i+1}: {entry['id']}")
        print(f"  HV: {entry['intent']['from_kv']}")
        print(f"  LV: {entry['intent']['to_kv']}")
        print(f"  MVA: {entry['intent']['rating_mva']}")
        print(f"  Scheme: {entry['intent']['scheme_hv']}")
        print(f"  Feeders: {entry['intent']['lv_feeders']}")
        print(f"  Protection: {entry['intent']['protection']}")
        print(f"  Sequence ({entry['sequence_length']} tokens):")
        
        # Show sequence in readable format
        tokens = entry["action_sequence"]
        sequence_str = " ".join(tokens[:15])  # First 15 tokens
        if len(tokens) > 15:
            sequence_str += " ... " + " ".join(tokens[-10:])  # Last 10 tokens
        print(f"    {sequence_str}")

if __name__ == "__main__":
    # Generate improved corpus
    print("🔧 Starting vocabulary-compliant corpus generation...")
    corpus = generate_corpus(1000)
    
    # Save corpus
    output_file = "improved_action_corpus.json"
    with open(output_file, 'w') as f:
        json.dump(corpus, f, indent=2)
    print(f"💾 Saved corpus to {output_file}")
    
    # Analyze results
    analyze_corpus(corpus)
    sample_corpus(corpus)
    
    print(f"\n✅ Vocabulary-compliant training corpus ready: {output_file}")
    print(f"   {len(corpus)} action sequences generated")
    print("   Perfect vocabulary compliance guaranteed!")
    print("   Ready for high-quality model training!")
