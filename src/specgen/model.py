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
from src.specgen.grammar_enforcer import GrammarEnforcer
from src.specgen.vocabulary import ActionVocabulary

@dataclass
class MinimalConfig:
    """Configuration for minimal model"""
    vocab_size: int = 324  # Updated to match actual vocabulary size
    d_model: int = 96
    n_heads: int = 2
    n_layers: int = 2
    max_seq_len: int = 200  # Increased to accommodate longer sequences
    dropout: float = 0.1
    learning_rate: float = 5e-4

    
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


class MinimalTransformer(nn.Module):
    """Tiny transformer for action sequence generation"""
    
    def __init__(self, config: MinimalConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        
        # Intent encoder (handles structured intent + grammar state)
        # Intent features: kv_in(11) + kv_out(11) + num_transformers(1) + mva(1) + percentZ(1) + kA(1) + thermal_A(1) + reliability(5) = 32
        # Grammar state features: basic_states(5) + inside_bracket(1) + current_action(10) + param_position(1) + seq_len(1) = 18
        # Total: 32 + 18 = 50
        self.intent_encoder = nn.Sequential(
            nn.Linear(50, config.d_model),  # Updated for intent + grammar state features
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
    
    def forward(self, intent_features: torch.Tensor, token_ids: torch.Tensor, 
               grammar_state_features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            intent_features: [batch_size, 32] - basic intent features
            token_ids: [batch_size, seq_len] - token sequence
            grammar_state_features: [batch_size, 18] - grammar state features (optional)
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = token_ids.shape
        
        # Combine intent and grammar state features
        if grammar_state_features is not None:
            combined_features = torch.cat([intent_features, grammar_state_features], dim=-1)
        else:
            # Fallback: pad with zeros if no grammar state provided
            zero_grammar = torch.zeros(batch_size, 18, device=intent_features.device)
            combined_features = torch.cat([intent_features, zero_grammar], dim=-1)
        
        # Encode combined features
        intent_embed = self.intent_encoder(combined_features)  # [batch_size, d_model]
        intent_embed = intent_embed.unsqueeze(1)  # [batch_size, 1, d_model]
        
        # Token embeddings
        token_embeds = self.token_embed(token_ids)  # [batch_size, seq_len, d_model]
        
        # Position embeddings
        positions = torch.arange(seq_len, device=token_ids.device)
        # Clamp positions to avoid out-of-bounds during generation
        positions = torch.clamp(positions, max=self.config.max_seq_len - 1)
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
        self.vocab = ActionVocabulary()
        self.model = MinimalTransformer(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.grammar = GrammarEnforcer(self.vocab)  # Add grammar enforcer
    
    def encode_grammar_state(self, sequence: List[str]) -> torch.Tensor:
        """Encode the current grammar state as a feature vector
        
        Args:
            sequence: Current token sequence
            
        Returns:
            Grammar state tensor with encoded features
        """
        # Get current grammar state from the enforcer
        current_state = self.grammar.get_current_state(sequence)
        
        # Define state encodings - these capture the structural context
        state_features = []
        
        # 1. Basic state encoding (one-hot for main states) - aligned with actual grammar
        basic_states = ['START', 'EXPECT_ACTION', 'EXPECT_PARAMS', 'ACTION_COMPLETE', 'COMPLETE']
        state_vector = [0.0] * len(basic_states)
        for i, state in enumerate(basic_states):
            if current_state == state or state in current_state:
                state_vector[i] = 1.0
        state_features.extend(state_vector)
        
        # 2. Inside list/bracket context (critical for CONNECT sequences)
        inside_bracket = 0.0
        bracket_count = 0
        if sequence:
            for token in sequence:
                if token == '[':
                    bracket_count += 1
                elif token == ']':
                    bracket_count -= 1
            inside_bracket = 1.0 if bracket_count > 0 else 0.0
        state_features.append(inside_bracket)
        
        # 3. Action context encoding
        current_action_features = [0.0] * 10  # 10 main actions
        actions = ['ADD_BUS', 'ADD_BAY', 'ADD_COUPLER', 'ADD_BREAKER', 'ADD_DISCONNECTOR', 
                  'ADD_TRANSFORMER', 'ADD_LINE', 'CONNECT', 'APPEND_TO_BAY', 'VALIDATE']
        
        # Find current action
        current_action = None
        if sequence:
            for i in range(len(sequence) - 1, -1, -1):
                if sequence[i] in actions:
                    current_action = sequence[i]
                    break
        
        if current_action:
            action_idx = actions.index(current_action)
            current_action_features[action_idx] = 1.0
        state_features.extend(current_action_features)
        
        # 4. Parameter position encoding (which parameter we're expecting)
        param_position = 0.0
        if current_action and current_action in self.grammar.action_param_sequences:
            expected_params = self.grammar.action_param_sequences[current_action]
            # Find action start position
            action_start_idx = -1
            for i in range(len(sequence) - 1, -1, -1):
                if sequence[i] == current_action:
                    action_start_idx = i
                    break
            
            if action_start_idx >= 0:
                tokens_since_action = sequence[action_start_idx + 1:]
                param_position = float(len(tokens_since_action)) / max(len(expected_params), 1)
        
        state_features.append(param_position)
        
        # 5. Sequence length encoding (normalized)
        seq_len_normalized = len(sequence) / 200.0  # Max sequence length
        state_features.append(seq_len_normalized)
        
        return torch.tensor(state_features, dtype=torch.float32)

    def extract_intent_features(self, intent: Dict) -> torch.Tensor:
        """Extract features from new structured intent format using vocabulary tokens"""
        
        # New intent structure using vocabulary tokens directly
        features = []
        
        # 1. HV voltage levels (kv_in) - encode as multi-hot vector
        kv_in_list = intent.get("kv_in", ["230kv"])
        voltage_vocab = ["345kv", "230kv", "138kv", "115kv", "69kv", "66kv", "46kv", "34.5kv", "25kv", "13.8kv", "4.16kv"]
        kv_in_vector = [0.0] * len(voltage_vocab)
        for voltage in kv_in_list:
            if voltage in voltage_vocab:
                kv_in_vector[voltage_vocab.index(voltage)] = 1.0
        features.extend(kv_in_vector)
        
        # 2. LV voltage levels (kv_out) - encode count per voltage
        kv_out_list = intent.get("kv_out", ["13.8kv", "13.8kv"])  # duplicates = feeder count
        kv_out_counts = [0.0] * len(voltage_vocab)
        for voltage in kv_out_list:
            if voltage in voltage_vocab:
                kv_out_counts[voltage_vocab.index(voltage)] += 1.0
        features.extend(kv_out_counts)
        
        # 3. Number of transformers - normalize count
        num_transformers = int(intent.get("num_transformers", "2"))
        features.append(float(num_transformers) / 10.0)  # Normalize to 0-1 range
        
        # 4. MVA rating - encode as index
        rated_mva = intent.get("rated_MVA", "150mva")
        mva_vocab = ["300mva", "150mva", "100mva", "75mva", "50mva", "25mva", "10mva", "5mva", "2.5mva", "1mva"]
        mva_index = float(mva_vocab.index(rated_mva)) if rated_mva in mva_vocab else 1.0
        features.append(mva_index / len(mva_vocab))  # Normalize
        
        # 5. Percent impedance - encode as index
        percent_z = intent.get("percentZ", "8percentZ")
        percentz_vocab = ["10percentZ", "8percentZ", "6percentZ", "4percentZ", "2percentZ", "1percentZ", "0.5percentZ"]
        percentz_index = float(percentz_vocab.index(percent_z)) if percent_z in percentz_vocab else 1.0
        features.append(percentz_index / len(percentz_vocab))  # Normalize
        
        # 6. HV interrupting current - encode as index
        hv_interrupting = intent.get("hv_interrupting_kA", "31.5kA")
        ka_vocab = ["63kA", "50kA", "40kA", "31.5kA", "25kA", "20kA", "16kA", "12.5kA", "10kA", "6.3kA", "4kA", "2.5kA"]
        ka_index = float(ka_vocab.index(hv_interrupting)) if hv_interrupting in ka_vocab else 3.0
        features.append(ka_index / len(ka_vocab))  # Normalize
        
        # 7. Feeder thermal ampacity - encode as index
        feeder_thermal = intent.get("feeder_thermal_A", "1200A")
        thermal_vocab = ["1200A", "1000A", "800A", "600A", "400A", "200A", "100A", "50A"]
        thermal_index = float(thermal_vocab.index(feeder_thermal)) if feeder_thermal in thermal_vocab else 0.0
        features.append(thermal_index / len(thermal_vocab))  # Normalize
        
        # 8. Reliability target - encode as categorical
        reliability = intent.get("reliability_target", "N_1")
        reliability_vocab = ["N_0", "N_1", "N_2", "REDUNDANT", "MINIMAL"]
        reliability_vector = [0.0] * len(reliability_vocab)
        if reliability in reliability_vocab:
            reliability_vector[reliability_vocab.index(reliability)] = 1.0
        else:
            reliability_vector[1] = 1.0  # Default to N_1
        features.extend(reliability_vector)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def compute_grammar_aware_loss(self, logits: torch.Tensor, targets: torch.Tensor, 
                                  sequences: List[List[str]], grammar_weight: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute loss with grammar penalty component enhanced by state encoding
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target token ids [batch_size, seq_len]  
            sequences: Token sequences for grammar validation [batch_size, seq_len]
            grammar_weight: Weight for grammar penalty term
            
        Returns:
            total_loss, cross_entropy_loss, grammar_penalty
        """
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            ignore_index=self.vocab.token_to_id.get("<PAD>", 0)
        )
        
        # Enhanced grammar penalty using state encoding
        grammar_penalty = 0.0
        batch_size, seq_len = targets.shape
        
        for b in range(batch_size):
            for t in range(seq_len):
                # Skip padding tokens
                if targets[b, t] == self.vocab.token_to_id.get("<PAD>", 0):
                    continue
                
                # Reconstruct sequence up to position t
                current_seq = []
                for i in range(t + 1):
                    token_id = targets[b, i].item()
                    if token_id < len(self.vocab.vocab):
                        current_seq.append(self.vocab.id_to_token[token_id])
                
                if not current_seq:
                    continue
                
                # Get grammar state for enhanced penalty weighting
                try:
                    grammar_state = self.encode_grammar_state(current_seq[:-1])  # Exclude target token
                    inside_bracket = grammar_state[5].item()  # Inside bracket feature
                    
                    # Get valid tokens using grammar enforcer
                    dummy_logits = torch.zeros(self.config.vocab_size)
                    masked_logits, valid_tokens, current_state = self.grammar.process_and_mask_for_generation(
                        current_seq[:-1], dummy_logits
                    )
                    
                    # Find which tokens are invalid
                    invalid_mask = torch.isinf(masked_logits)
                    
                    # Get prediction probabilities
                    pred_probs = F.softmax(logits[b, t], dim=-1)
                    
                    # Calculate base penalty
                    invalid_prob_mass = torch.sum(pred_probs[invalid_mask])
                    
                    # Apply enhanced penalty based on grammar state
                    penalty_multiplier = 1.0
                    
                    # Higher penalty when inside brackets (stricter ID token enforcement)
                    if inside_bracket > 0.5:
                        penalty_multiplier *= 3.0  # Much higher penalty inside brackets
                        
                        # Extra penalty for non-ID tokens when inside brackets
                        id_tokens = set()
                        for token_set_name in ['bus_ids', 'breaker_ids', 'coupler_ids', 'bay_ids', 
                                             'disconnector_ids', 'transformer_ids', 'line_ids']:
                            if token_set_name in self.vocab.token_sets:
                                id_tokens.update(self.vocab.token_sets[token_set_name])
                        id_tokens.update({',', ']'})  # Allow comma and closing bracket
                        
                        # Check if predicted token is not an ID token
                        predicted_token_id = torch.argmax(pred_probs).item()
                        if predicted_token_id < len(self.vocab.vocab):
                            predicted_token = self.vocab.id_to_token[predicted_token_id]
                            if predicted_token not in id_tokens:
                                penalty_multiplier *= 5.0  # Severe penalty for wrong token types in brackets
                    
                    # Apply context-sensitive penalty
                    grammar_penalty += penalty_multiplier * invalid_prob_mass
                    
                except Exception:
                    # Fallback to standard penalty if state processing fails
                    continue
        
        # Normalize by total predictions
        if batch_size * seq_len > 0:
            grammar_penalty = grammar_penalty / (batch_size * seq_len)
        
        # Combine losses
        total_loss = ce_loss + grammar_weight * grammar_penalty
        
        return total_loss, ce_loss, grammar_penalty
    
    def load_training_data(self, data_file: str = "data/dev_data.json") -> List[Tuple[torch.Tensor, List[str]]]:
        """Load training data from external file"""
        import os
        
        # Get the project root directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.join(current_dir, '..', '..')
        data_path = os.path.join(project_root, data_file)
        
        training_examples = []
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            for example in data:
                intent = example["intent"]
                sequence = example["sequence"]
                
                # Extract intent features using the new format
                intent_features = self.extract_intent_features(intent)
                training_examples.append((intent_features, sequence))
                
            print(f"Loaded {len(training_examples)} training examples from {data_file}")
            
        except FileNotFoundError:
            print(f"Warning: Training data file {data_path} not found. Using empty dataset.")
            return []
        except Exception as e:
            print(f"Error loading training data: {e}")
            return []
            
        return training_examples
    
    def train(self, data_file: str = "data/dev_data.json", num_epochs: int = 10, 
              use_grammar_loss: bool = True, grammar_weight: float = 0.1):
        """Training loop using external data file with optional grammar-aware loss
        
        Args:
            data_file: Path to training data
            num_epochs: Number of training epochs
            use_grammar_loss: Whether to use grammar penalty in loss
            grammar_weight: Weight for grammar penalty component
        """
        print(f"Loading training data from {data_file}...")
        training_data = self.load_training_data(data_file)
        
        if not training_data:
            print("No training data available. Cannot train model.")
            return
        
        print(f"Training for {num_epochs} epochs...")
        if use_grammar_loss:
            print(f"Using grammar-aware loss with weight {grammar_weight}")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_total_loss = 0.0
            epoch_ce_loss = 0.0
            epoch_grammar_penalty = 0.0
            
            # Shuffle data
            random.shuffle(training_data)
            
            for intent_features, tokens in training_data:
                # Use unified grammar enforcer method to check, repair, and convert to indices
                token_ids, is_valid, error_message = self.grammar.process_sequence_for_training(
                    tokens, repair_if_invalid=True
                )
                
                # Log any repair actions for debugging
                if not is_valid:
                    print(f"Warning: Invalid sequence processed - {error_message}")
                
                # Skip sequences that are too long after processing
                if len(token_ids) > self.config.max_seq_len:
                    continue
                    
                # Prepare tensors
                intent_batch = intent_features.unsqueeze(0)  # Add batch dimension
                input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long)  # All but last
                target_ids = torch.tensor([token_ids[1:]], dtype=torch.long)   # All but first
                
                # Compute grammar state features for input sequence
                # Convert input token ids back to tokens for grammar state computation
                input_tokens = self.vocab.decode(input_ids[0].tolist())
                grammar_state = self.encode_grammar_state(["<START>"] + input_tokens)
                grammar_state_batch = grammar_state.unsqueeze(0)  # Add batch dimension
                
                # Forward pass with grammar state
                logits = self.model(intent_batch, input_ids, grammar_state_batch)
                
                # Compute loss (grammar-aware or standard)
                if use_grammar_loss:
                    # Convert target ids back to token sequences for grammar checking
                    target_sequences = [self.vocab.decode(target_ids[0].tolist())]
                    total_loss, ce_loss, grammar_penalty = self.compute_grammar_aware_loss(
                        logits, target_ids, target_sequences, grammar_weight
                    )
                    
                    epoch_ce_loss += ce_loss.item()
                    epoch_grammar_penalty += grammar_penalty.item()
                else:
                    # Standard cross-entropy loss
                    total_loss = F.cross_entropy(
                        logits.reshape(-1, self.config.vocab_size),
                        target_ids.reshape(-1)
                    )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                epoch_total_loss += total_loss.item()
            
            # Print epoch statistics
            avg_total_loss = epoch_total_loss / len(training_data)
            if use_grammar_loss:
                avg_ce_loss = epoch_ce_loss / len(training_data)
                avg_grammar_penalty = epoch_grammar_penalty / len(training_data)
                print(f"Epoch {epoch+1}/{num_epochs}: "
                      f"Total Loss = {avg_total_loss:.4f}, "
                      f"CE Loss = {avg_ce_loss:.4f}, "
                      f"Grammar Penalty = {avg_grammar_penalty:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_total_loss:.4f}")
    
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
                start_id = self.vocab.encode(["<START>"])[0]
                current_tokens = [start_id]
                current_sequence = ["<START>"]
                
                for step in range(max_length):
                    # Prepare input
                    input_ids = torch.tensor([current_tokens], dtype=torch.long)
                    
                    # Compute grammar state for current sequence
                    grammar_state = self.encode_grammar_state(current_sequence)
                    grammar_state_batch = grammar_state.unsqueeze(0)  # Add batch dimension
                    
                    # Forward pass with grammar state
                    logits = self.model(intent_features, input_ids, grammar_state_batch)
                    last_logits = logits[0, -1, :]  # [vocab_size]
                    
                    # Apply unified grammar processing: check sequence and mask invalid tokens
                    masked_logits, valid_tokens, current_state = self.grammar.process_and_mask_for_generation(
                        current_sequence, last_logits
                    )
                    
                    # Apply temperature and sample
                    if temperature > 0:
                        # Sample with temperature
                        probs = F.softmax(masked_logits / temperature, dim=-1)
                        next_token_id = torch.multinomial(probs, 1).item()
                    else:
                        # Greedy (deterministic)
                        next_token_id = torch.argmax(masked_logits).item()
                    
                    next_token = self.vocab.decode([next_token_id])[0]
                    
                    # Check for end token
                    if next_token == "<END>":
                        break
                    
                    # Add to sequence
                    current_tokens.append(next_token_id)
                    current_sequence.append(next_token)
                
                # Decode tokens (remove START)
                tokens = current_sequence[1:]  # Remove START token
                
                # Apply final grammar repair and validation using unified method
                full_sequence_with_end = ["<START>"] + tokens + ["<END>"]
                final_token_ids, is_complete, repair_message = self.grammar.process_sequence_for_training(
                    full_sequence_with_end, repair_if_invalid=True
                )
                
                # Extract tokens (remove START and END)
                if is_complete:
                    final_tokens = self.vocab.decode(final_token_ids[1:-1])
                    tokens = [t for t in final_tokens if t not in ["<END>", "<PAD>", "<UNK>"]]
                else:
                    # Use repaired tokens if available
                    repaired_tokens = self.vocab.decode(final_token_ids)
                    tokens = [t for t in repaired_tokens[1:-1] if t not in ["<END>", "<PAD>", "<UNK>"]]
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
    
    print("Training minimal action-token model with external data...")
    model.train(data_file="data/dev_data.json", num_epochs=20)
    
    # Test generation with new intent format
    test_intent = {
        "kv_in": ["230kv"],
        "kv_out": ["13.8kv", "13.8kv", "13.8kv"],
        "num_transformers": "1",
        "rated_MVA": "150mva", 
        "percentZ": "8percentZ",
        "hv_interrupting_kA": "31.5kA",
        "feeder_thermal_A": "1200A",
        "reliability_target": "N_1"
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