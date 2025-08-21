"""
minimal_model.py

Minimal viable action-token language model for electrical substation generation.
Demonstrates core concept with smallest possible implementation.

Author: Gordon Doore
Date Created: 2025-08-19
Last Modified: 2025-08-21
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from scripts.data.data_generator import ElectricalDataGenerator

@dataclass
class MinimalConfig:
    """Configuration for minimal model"""
    vocab_size: int = 200  # Will be set from vocabulary size
    d_model: int = 96
    n_heads: int = 2
    n_layers: int = 2
    max_seq_len: int = 200  # Increased to accommodate longer sequences
    dropout: float = 0.1
    learning_rate: float = 5e-4

class GrammarEnforcer:
    """Enforces syntactic grammar for electrical substation action sequences"""
    
    def __init__(self, vocab: 'MinimalVocabulary'):
        self.vocab = vocab
        self._build_grammar_rules()
    
    def _build_grammar_rules(self):
        """Define valid token transitions and argument patterns"""
        
        # Valid action commands
        self.actions = {
            "ADD_BUS", "ADD_TRANSFORMER", "ADD_BAY", "APPEND_STEP", 
            "CONNECT_BUS", "CONNECT_TX_HV", "CONNECT_TX_LV", "EMIT_SPEC"
        }
        
        # Valid voltage tokens
        self.voltages = {
            "345KV", "230KV", "220KV", "138KV", "115KV", "69KV", "66KV", 
            "46KV", "34.5KV", "25KV", "13.8KV", "4.16KV"
        }
        
        # Valid MVA tokens
        self.mva_ratings = {
            "300MVA", "150MVA", "100MVA", "75MVA", "50MVA", "25MVA", "10MVA"
        }
        
        # Valid bus/component roles
        self.roles = {"HV", "LV", "MAIN"}
        
        # Valid bay types
        self.bay_types = {"TRANSFORMER_BAY", "FEEDER_BAY"}
        
        # Valid device types for APPEND_STEP
        self.devices = {"BREAKER", "BUS_ISOLATOR", "LINE_ISOLATOR", "CT"}
        
        # Valid transformer types
        self.transformer_types = {"TWO_WINDING"}
        
        # Valid semantic IDs
        self.bus_ids = {f"BUS_{i}" for i in range(101, 121)}
        self.tx_ids = {f"TX_{i}" for i in range(201, 211)}
        self.bay_ids = {f"BAY_{i}" for i in range(301, 341)}
        
        # Grammar patterns for each action
        self.action_patterns = {
            "ADD_BUS": [self.voltages, self.roles, {"MAIN"}, self.bus_ids],
            "ADD_TRANSFORMER": [self.transformer_types, self.voltages, self.voltages, self.mva_ratings, self.tx_ids],
            "ADD_BAY": [self.roles, self.voltages, self.bay_types, self.bay_ids],
            "APPEND_STEP": [self.bay_ids, self.devices],
            "CONNECT_BUS": [self.bus_ids, self.bay_ids],
            "CONNECT_TX_HV": [self.tx_ids, self.bus_ids],
            "CONNECT_TX_LV": [self.tx_ids, self.bus_ids],
        }
        
        # Special tokens
        self.special_tokens = {"<START>", "<END>", "<PAD>", "<UNK>"}
    
    def get_valid_next_tokens(self, sequence: List[str]) -> List[str]:
        """Get list of valid next tokens given current sequence"""
        if not sequence:
            return ["<START>"]
        
        if sequence[-1] == "<START>":
            # After START, only actions are valid
            return list(self.actions)
        
        if sequence[-1] in self.actions:
            # After an action, need the first argument
            action = sequence[-1]
            if action in self.action_patterns:
                return list(self.action_patterns[action][0])
            return []
        
        # Find the current action context
        current_action = None
        arg_position = -1
        
        for i in range(len(sequence) - 1, -1, -1):
            if sequence[i] in self.actions:
                current_action = sequence[i]
                arg_position = len(sequence) - i - 1
                break
        
        if current_action and current_action in self.action_patterns:
            pattern = self.action_patterns[current_action]
            if arg_position < len(pattern):
                # Still expecting arguments for current action
                return list(pattern[arg_position])
            else:
                # Action complete, can start new action or end
                return list(self.actions) + ["<END>"]
        
        # Default: can start new action or end
        return list(self.actions) + ["<END>"]
    
    def mask_invalid_tokens(self, logits: torch.Tensor, sequence: List[str]) -> torch.Tensor:
        """Apply grammar mask to logits, setting invalid tokens to -inf"""
        valid_tokens = self.get_valid_next_tokens(sequence)
        valid_ids = [self.vocab.token_to_id.get(token, self.vocab.token_to_id["<UNK>"]) for token in valid_tokens]
        
        # Create mask
        mask = torch.full_like(logits, float('-inf'))
        mask[valid_ids] = 0
        
        return logits + mask
    
    def is_sequence_complete(self, sequence: List[str]) -> bool:
        """Check if sequence is grammatically complete"""
        if not sequence or sequence[-1] != "<END>":
            return False
        
        # Must start with <START>
        if sequence[0] != "<START>":
            return False
        
        # Check each action is complete
        i = 1  # Skip <START>
        while i < len(sequence) - 1:  # Skip <END>
            if sequence[i] in self.actions:
                action = sequence[i]
                if action in self.action_patterns:
                    pattern = self.action_patterns[action]
                    # Check all arguments are present
                    if i + len(pattern) >= len(sequence) - 1:
                        return False
                    # Check argument validity
                    for j, expected_set in enumerate(pattern):
                        if sequence[i + 1 + j] not in expected_set:
                            return False
                    i += len(pattern) + 1
                else:
                    return False
            else:
                return False
        
        return True
    
    def repair_sequence(self, sequence: List[str]) -> List[str]:
        """Attempt to repair an invalid sequence"""
        if not sequence:
            return ["<START>", "<END>"]
        
        repaired = []
        i = 0
        
        # Ensure starts with <START>
        if sequence[0] != "<START>":
            repaired.append("<START>")
        else:
            repaired.append("<START>")
            i = 1  # Skip the <START> token we just added
        
        while i < len(sequence):
            token = sequence[i]
            
            if token in self.actions:
                repaired.append(token)
                # Add required arguments
                if token in self.action_patterns:
                    pattern = self.action_patterns[token]
                    for j, expected_set in enumerate(pattern):
                        if i + 1 + j < len(sequence) and sequence[i + 1 + j] in expected_set:
                            repaired.append(sequence[i + 1 + j])
                        else:
                            # Use first valid token from set
                            repaired.append(list(expected_set)[0])
                    i += len(pattern) + 1
                else:
                    i += 1
            elif token == "<END>":
                repaired.append(token)
                break
            elif token in self.special_tokens and token != "<START>":
                # Skip other special tokens except END
                i += 1
            else:
                i += 1
        
        # Ensure ends with <END>
        if not repaired or repaired[-1] != "<END>":
            repaired.append("<END>")
        
        return repaired

class MinimalVocabulary:
    """Simplified vocabulary for proof of concept"""
    
    def __init__(self):
        self.vocab = self._create_vocab()
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
    
    def _create_vocab(self) -> List[str]:
        """Create minimal vocabulary focusing on most common tokens"""
        vocab = [
            # Special tokens
            "<START>", "<END>", "<PAD>", "<UNK>",
            
            # Core actions
            "ADD_BUS", "ADD_TRANSFORMER", "ADD_BAY", "APPEND_STEP", 
            "ADD_CONNECTION", "EMIT_SPEC",
            
            # Common parameters
            "345KV", "230KV", "220KV", "138KV", "115KV", "69KV", "66KV", "46KV","34.5KV", "25KV", "13.8KV", "4.16KV",
            "300MVA", "150MVA", "100MVA", "75MVA", "50MVA", "25MVA", "10MVA",
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
            
            # Note: Devices (up to 100) don't need explicit IDs as they are ordered lists within bays
        ]
        
        # Pad to vocab_size if needed
        while len(vocab) < 120:
            vocab.append(f"<RESERVED_{len(vocab)}>")
        
        return vocab[:120]  # Limit to exact vocab size
    
    def encode(self, tokens: List[str]) -> List[int]:
        """Convert tokens to IDs"""
        return [self.token_to_id.get(token, self.token_to_id["<UNK>"]) for token in tokens]
    
    def decode(self, ids: List[int]) -> List[str]:
        """Convert IDs to tokens"""
        return [self.id_to_token.get(id, "<UNK>") for id in ids]

class MinimalTransformer(nn.Module):
    """Tiny transformer for action sequence generation"""
    
    def __init__(self, config: MinimalConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Intent encoder (simple MLP)
        self.intent_encoder = nn.Sequential(
            nn.Linear(6, config.d_model),  # 6 basic intent features
            nn.ReLU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.d_model * 2,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, intent_features: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            intent_features: [batch_size, 6] - basic intent features
            token_ids: [batch_size, seq_len] - token sequence
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = token_ids.shape
        
        # Encode intent
        intent_embed = self.intent_encoder(intent_features)  # [batch_size, d_model]
        intent_embed = intent_embed.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Token embeddings
        token_embeds = self.token_embed(token_ids)  # [batch_size, seq_len, d_model]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=token_ids.device)
        pos_embeds = self.pos_embed(positions).unsqueeze(0)  # [1, seq_len, d_model]
        
        # Combine embeddings
        x = self.dropout(token_embeds + pos_embeds)  # [batch_size, seq_len, d_model]
        
        # Causal mask
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len)
        causal_mask = causal_mask.to(token_ids.device)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(tgt=x, memory=intent_embed, tgt_mask=causal_mask)
        
        # Project to vocabulary
        logits = self.output_proj(x)  # [batch_size, seq_len, vocab_size]
        
        return logits

class MinimalActionModel:
    """Complete minimal model with training and inference"""
    
    def __init__(self, config: MinimalConfig):
        self.config = config
        self.vocab = MinimalVocabulary()
        self.model = MinimalTransformer(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.grammar = GrammarEnforcer(self.vocab)  # Add grammar enforcer
    
    def extract_intent_features(self, intent: Dict) -> torch.Tensor:
        """Extract enhanced features from intent JSON with explicit voltage mapping"""
        
        # Map voltages to their vocab token indices for better learning
        voltage_vocab = ["345KV", "220KV", "138KV", "115KV", "69KV", "66KV", "34.5KV", "25KV", "13.8KV", "4.16KV"]
        
        def voltage_to_index(kv_value):
            """Map voltage value to vocabulary index"""
            kv_str = f"{kv_value}KV"
            if kv_str in voltage_vocab:
                return float(voltage_vocab.index(kv_str))
            # Find closest match
            closest_idx = 0
            closest_diff = float('inf')
            for i, vocab_kv in enumerate(voltage_vocab):
                vocab_val = float(vocab_kv.replace("KV", ""))
                diff = abs(vocab_val - kv_value)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_idx = i
            return float(closest_idx)
        
        # Map MVA ratings to indices
        mva_vocab = ["300MVA", "150MVA", "100MVA", "75MVA", "50MVA", "25MVA", "10MVA"]
        
        def mva_to_index(mva_value):
            """Map MVA value to vocabulary index"""
            mva_str = f"{int(mva_value)}MVA"
            if mva_str in mva_vocab:
                return float(mva_vocab.index(mva_str))
            # Find closest match
            closest_idx = 0
            closest_diff = float('inf')
            for i, vocab_mva in enumerate(mva_vocab):
                vocab_val = int(vocab_mva.replace("MVA", ""))
                diff = abs(vocab_val - mva_value)
                if diff < closest_diff:
                    closest_diff = diff
                    closest_idx = i
            return float(closest_idx)
        
        features = [
            voltage_to_index(float(intent.get("from_kv", 220))),  # HV voltage index
            voltage_to_index(float(intent.get("to_kv", 66))),     # LV voltage index
            mva_to_index(float(intent.get("rating_mva", 100))),   # MVA rating index
            float(intent.get("lv_feeders", 2)),                   # Number of feeders
            1.0 if intent.get("scheme_hv") == "double_busbar" else 0.0,
            1.0 if intent.get("scheme_lv") == "radial" else 0.0
        ]
        return torch.tensor(features, dtype=torch.float32)
    
    def create_training_example(self, intent: Dict) -> Tuple[torch.Tensor, List[str]]:
        """Create single training example from intent with design variations"""
        # Simple canonical sequence based on intent
        tokens = ["<START>"]
        
        # Add buses using semantic IDs (with some variation in approach)
        if intent.get("scheme_hv") == "double_busbar":
            tokens.extend([
                "ADD_BUS", f"{intent['from_kv']}KV", "HV", "MAIN", "BUS_101",
                "ADD_BUS", f"{intent['from_kv']}KV", "HV", "MAIN", "BUS_102"
            ])
            main_bus = "BUS_101"  # Primary bus for connections
        else:
            tokens.extend([
                "ADD_BUS", f"{intent['from_kv']}KV", "HV", "MAIN", "BUS_101"
            ])
            main_bus = "BUS_101"
        
        # Add LV bus
        tokens.extend([
            "ADD_BUS", f"{intent['to_kv']}KV", "LV", "MAIN", "BUS_103"
        ])
        
        # Add transformer
        tokens.extend([
            "ADD_TRANSFORMER", "TWO_WINDING", f"{intent['from_kv']}KV", 
            f"{intent['to_kv']}KV", f"{intent['rating_mva']}MVA", "TX_201"
        ])
        
        # Add transformer bay with variation based on protection level
        protection_level = intent.get("protection", "standard")
        tokens.extend([
            "ADD_BAY", "HV", f"{intent['from_kv']}KV", "TRANSFORMER_BAY", "BAY_301"
        ])
        
        # Vary protection sequence based on design philosophy
        if protection_level == "enhanced":
            tokens.extend([
                "APPEND_STEP", "BAY_301", "BUS_ISOLATOR",
                "APPEND_STEP", "BAY_301", "CT",  # Current transformer for enhanced protection
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
        
        # Add feeders with some variation
        num_feeders = int(intent.get("lv_feeders", 2))
        for i in range(num_feeders):
            bay_id = f"BAY_{302+i}"
            tokens.extend([
                "ADD_BAY", "LV", f"{intent['to_kv']}KV", "FEEDER_BAY", bay_id
            ])
            
            # Vary feeder protection based on importance/size
            if i == 0 or protection_level == "enhanced":  # First feeder or enhanced gets more protection
                tokens.extend([
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR",
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "CT"
                ])
            else:  # Subsequent feeders get basic protection
                tokens.extend([
                    "APPEND_STEP", bay_id, "BREAKER",
                    "APPEND_STEP", bay_id, "BUS_ISOLATOR"
                ])
        
        # Add connections
        tokens.extend([
            "CONNECT_BUS", main_bus, "BAY_301",
            "CONNECT_TX_HV", "TX_201", main_bus,
            "CONNECT_TX_LV", "TX_201", "BUS_103"
        ])
        
        tokens.append("<END>")
        
        # Clean up tokens to match vocabulary
        cleaned_tokens = []
        for token in tokens:
            if token in self.vocab.token_to_id:
                cleaned_tokens.append(token)
            else:
                # Map unknown tokens to closest match
                if token.endswith("KV"):
                    # Map voltage to closest vocabulary voltage
                    if "345" in token:
                        cleaned_tokens.append("345KV")
                    elif "220" in token:
                        cleaned_tokens.append("220KV")
                    elif "138" in token:
                        cleaned_tokens.append("138KV")
                    elif "115" in token:
                        cleaned_tokens.append("115KV")
                    elif "69" in token:
                        cleaned_tokens.append("69KV")
                    elif "66" in token:
                        cleaned_tokens.append("66KV")
                    elif "34.5" in token or "34_5" in token:
                        cleaned_tokens.append("34.5KV")
                    elif "25" in token:
                        cleaned_tokens.append("25KV")
                    elif "13.8" in token:
                        cleaned_tokens.append("13.8KV")
                    else:
                        cleaned_tokens.append("220KV")  # Default
                elif token.endswith("MVA"):
                    # Map MVA to closest vocabulary MVA
                    if "150" in token:
                        cleaned_tokens.append("150MVA")
                    elif "100" in token:
                        cleaned_tokens.append("100MVA")
                    elif "75" in token:
                        cleaned_tokens.append("75MVA")
                    elif "50" in token:
                        cleaned_tokens.append("50MVA")
                    else:
                        cleaned_tokens.append("100MVA")  # Default
                else:
                    cleaned_tokens.append("<UNK>")
        
        intent_features = self.extract_intent_features(intent)
        return intent_features, cleaned_tokens
    
    def generate_training_data(self, num_examples: int = 100, use_realistic_data: bool = True) -> List[Tuple[torch.Tensor, List[str]]]:
        """Generate synthetic training dataset with option for realistic data"""
        
        if use_realistic_data:
            # Use the advanced data generator
            
            generator = ElectricalDataGenerator()
            realistic_dataset = generator.generate_training_dataset(num_examples)
            
            training_data = []
            for example in realistic_dataset:
                intent = example["intent"]
                features, tokens = self.create_training_example(intent)
                training_data.append((features, tokens))
            
            return training_data
        
        else:
            # Fall back to simple synthetic data (original method)
            # Expanded base intent variations for more diversity
            base_intents = [
                # Standard configurations
                {"from_kv": 220, "to_kv": 66, "rating_mva": 100, "scheme_hv": "double_busbar", "lv_feeders": 2, "protection": "standard"},
                {"from_kv": 138, "to_kv": 13.8, "rating_mva": 75, "scheme_hv": "single_busbar", "lv_feeders": 3, "protection": "standard"},
                {"from_kv": 69, "to_kv": 13.8, "rating_mva": 50, "scheme_hv": "double_busbar", "lv_feeders": 4, "protection": "standard"},
                
                # High-reliability variations
                {"from_kv": 220, "to_kv": 66, "rating_mva": 100, "scheme_hv": "double_busbar", "lv_feeders": 2, "protection": "enhanced"},
                {"from_kv": 138, "to_kv": 66, "rating_mva": 150, "scheme_hv": "double_busbar", "lv_feeders": 6, "protection": "enhanced"},
                
                # Compact/cost-optimized variations
                {"from_kv": 138, "to_kv": 25, "rating_mva": 50, "scheme_hv": "single_busbar", "lv_feeders": 2, "protection": "basic"},
                {"from_kv": 69, "to_kv": 13.8, "rating_mva": 75, "scheme_hv": "single_busbar", "lv_feeders": 3, "protection": "basic"},
                
                # Industrial/special purpose
                {"from_kv": 345, "to_kv": 138, "rating_mva": 150, "scheme_hv": "double_busbar", "lv_feeders": 8, "protection": "enhanced"},
                {"from_kv": 115, "to_kv": 34.5, "rating_mva": 75, "scheme_hv": "single_busbar", "lv_feeders": 4, "protection": "standard"},
            ]
            
            training_data = []
            
            for i in range(num_examples):
                # Pick base intent and add variations
                intent = base_intents[i % len(base_intents)].copy()
                
                # Add controlled randomness for variety
                intent["lv_feeders"] = random.randint(max(2, intent["lv_feeders"]-1), intent["lv_feeders"]+2)
                intent["lv_feeders"] = min(intent["lv_feeders"], 8)  # Cap at 8 feeders
                
                # Randomly vary some design choices
                if random.random() < 0.3:  # 30% chance to flip busbar scheme
                    intent["scheme_hv"] = "single_busbar" if intent["scheme_hv"] == "double_busbar" else "double_busbar"
                
                # Create training example
                features, tokens = self.create_training_example(intent)
                training_data.append((features, tokens))
            
            return training_data
    
    def validate_training_sequence(self, tokens: List[str]) -> List[str]:
        """Validate and repair training sequence using grammar rules"""
        if not tokens:
            return ["<START>", "<END>"]
        
        # Ensure proper start/end tokens
        if tokens[0] != "<START>":
            tokens = ["<START>"] + tokens
        if tokens[-1] != "<END>":
            tokens = tokens + ["<END>"]
        
        # Apply grammar repair
        if not self.grammar.is_sequence_complete(tokens):
            tokens = self.grammar.repair_sequence(tokens)
        
        return tokens
    
    def train(self, num_examples: int = 100, num_epochs: int = 10):
        """Simple training loop"""
        print(f"Generating {num_examples} training examples...")
        training_data = self.generate_training_data(num_examples)
        
        print(f"Training for {num_epochs} epochs...")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            
            # Shuffle data
            random.shuffle(training_data)
            
            for intent_features, tokens in training_data:
                # Validate sequence grammar
                validated_tokens = self.validate_training_sequence(tokens)
                
                # Encode tokens
                token_ids = self.vocab.encode(validated_tokens)
                
                # Skip sequences that are too long
                if len(token_ids) > self.config.max_seq_len:
                    continue
                
                # Prepare tensors
                intent_batch = intent_features.unsqueeze(0)  # Add batch dimension
                input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long)  # All but last
                target_ids = torch.tensor([token_ids[1:]], dtype=torch.long)   # All but first
                
                # Forward pass
                logits = self.model(intent_batch, input_ids)
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.reshape(-1, self.config.vocab_size),
                    target_ids.reshape(-1)
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(training_data)
            print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
    
    def generate(self, intent: Dict, max_length: int = 200, temperature: float = 0.8, num_candidates: int = 1) -> List[List[str]]:
        """Generate action sequence(s) from intent with grammar enforcement
        
        Args:
            intent: Intent specification
            max_length: Maximum sequence length
            temperature: Controls randomness (0.0=deterministic, 1.0=very random)
            num_candidates: Number of different options to generate
            
        Returns:
            List of token sequences (one per candidate)
        """
        self.model.eval()
        candidates = []
        
        for _ in range(num_candidates):
            with torch.no_grad():
                # Extract intent features
                intent_features = self.extract_intent_features(intent).unsqueeze(0)
                
                # Start with START token
                current_tokens = [self.vocab.token_to_id["<START>"]]
                current_sequence = ["<START>"]
                
                for step in range(max_length):
                    # Prepare input
                    input_ids = torch.tensor([current_tokens], dtype=torch.long)
                    
                    # Forward pass
                    logits = self.model(intent_features, input_ids)
                    last_logits = logits[0, -1, :]  # [vocab_size]
                    
                    # Apply grammar mask to enforce valid next tokens
                    masked_logits = self.grammar.mask_invalid_tokens(last_logits, current_sequence)
                    
                    # Apply temperature and sample
                    if temperature > 0:
                        # Sample with temperature
                        probs = F.softmax(masked_logits / temperature, dim=-1)
                        next_token_id = torch.multinomial(probs, 1).item()
                    else:
                        # Greedy (deterministic)
                        next_token_id = torch.argmax(masked_logits).item()
                    
                    next_token = self.vocab.id_to_token[next_token_id]
                    
                    # Check for end token
                    if next_token == "<END>":
                        break
                    
                    # Add to sequence
                    current_tokens.append(next_token_id)
                    current_sequence.append(next_token)
                
                # Decode tokens (remove START)
                tokens = current_sequence[1:]  # Remove START token
                
                # Apply final grammar repair if needed
                full_sequence = ["<START>"] + tokens + ["<END>"]
                if not self.grammar.is_sequence_complete(full_sequence):
                    repaired = self.grammar.repair_sequence(full_sequence)
                    tokens = repaired[1:-1]  # Remove START and END
                
                # Remove special tokens
                tokens = [t for t in tokens if t not in ["<END>", "<PAD>", "<UNK>"]]
                candidates.append(tokens)
        
        return candidates
    
    def generate_from_intent(self, intent: Dict, design_philosophy: str = "standard", 
                           temperature: float = 0.7, max_length: int = 200) -> List[str]:
        """Generate single action sequence from intent for validation
        
        Args:
            intent: Intent specification dict
            design_philosophy: "conservative", "standard", or "economical" 
            temperature: Sampling temperature
            max_length: Maximum sequence length
            
        Returns:
            Single action sequence as list of tokens
        """
        # Modify intent based on design philosophy
        modified_intent = intent.copy()
        
        if design_philosophy == "conservative":
            modified_intent["protection"] = "enhanced"
            modified_intent["scheme_hv"] = "double_busbar"
        elif design_philosophy == "economical":
            modified_intent["protection"] = "basic"
            modified_intent["scheme_hv"] = "single_busbar"
        # standard uses intent as-is
        
        # Generate single candidate
        candidates = self.generate(
            modified_intent, 
            max_length=max_length, 
            temperature=temperature, 
            num_candidates=1
        )
        
        return candidates[0] if candidates else []
    
    def generate_design_options(self, base_intent: Dict) -> Dict[str, List[str]]:
        """Generate different design options for the same base requirements
        
        Returns:
            Dictionary with design approach names and their token sequences
        """
        options = {}
        
        # Conservative/High-reliability option
        conservative_intent = base_intent.copy()
        conservative_intent["protection"] = "enhanced"
        conservative_intent["scheme_hv"] = "double_busbar"  # Force double busbar for reliability
        options["conservative"] = self.generate(conservative_intent, temperature=0.2, num_candidates=1)[0]
        
        # Standard/Balanced option
        standard_intent = base_intent.copy()
        standard_intent["protection"] = "standard"
        options["standard"] = self.generate(standard_intent, temperature=0.5, num_candidates=1)[0]
        
        # Economical/Minimal option
        economical_intent = base_intent.copy()
        economical_intent["protection"] = "basic"
        economical_intent["scheme_hv"] = "single_busbar"  # Force single busbar for cost savings
        economical_intent["lv_feeders"] = max(2, economical_intent.get("lv_feeders", 2) - 1)  # Reduce feeders if possible
        options["economical"] = self.generate(economical_intent, temperature=0.3, num_candidates=1)[0]
        
        return options

def demo_minimal_model():
    """Demonstration of minimal model"""
    
    # Create and train model
    config = MinimalConfig()
    model = MinimalActionModel(config)
    
    print("Training minimal action-token model with realistic data...")
    model.train(num_examples=500, num_epochs=20)  # Use realistic data
    
    # Test generation
    test_intent = {
        "from_kv": 220,
        "to_kv": 66,
        "rating_mva": 100,
        "scheme_hv": "double_busbar",
        "lv_feeders": 3
    }
    
    print("\nGenerating sequence for intent:")
    print(json.dumps(test_intent, indent=2))
    
    # Generate multiple options with different temperatures
    print("\n=== OPTION 1: Conservative Design (low temperature) ===")
    conservative_options = model.generate(test_intent, temperature=0.3, num_candidates=1)
    print(" ".join(conservative_options[0]))
    
    print("\n=== OPTION 2: Standard Design (medium temperature) ===")
    standard_options = model.generate(test_intent, temperature=0.8, num_candidates=1)
    print(" ".join(standard_options[0]))
    
    print("\n=== OPTION 3: Creative Design (high temperature) ===")
    creative_options = model.generate(test_intent, temperature=1.5, num_candidates=1)
    print(" ".join(creative_options[0]))
    
    print("\n=== OPTION 4: Very Creative Design (very high temperature) ===")
    very_creative_options = model.generate(test_intent, temperature=2.0, num_candidates=1)
    print(" ".join(very_creative_options[0]))
    
    print("\n=== MULTIPLE OPTIONS (3 candidates at medium temperature) ===")
    multiple_options = model.generate(test_intent, temperature=0.8, num_candidates=3)
    for i, option in enumerate(multiple_options, 1):
        print(f"Candidate {i}: {' '.join(option)}")
    
    return model

if __name__ == "__main__":
    demo_minimal_model()