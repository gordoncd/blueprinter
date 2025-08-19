"""
corpus_trainer.py

Training script that uses the generated action sequence corpus.
This replaces the synthetic data generation with real corpus-based training.

Author: Gordon Doore
Date Created: 2025-08-19
"""

import json
import torch
import random
import sys
import os
from typing import List, Tuple

# Add the src directory to the path so we can import from specgen
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.specgen.model import MinimalActionModel, MinimalConfig

class CorpusTrainer:
    """Train model using pre-generated action sequence corpus"""
    
    def __init__(self, model: MinimalActionModel):
        self.model = model
        
    def load_corpus(self, corpus_file: str) -> List[dict]:
        """Load the action sequence corpus"""
        print(f"Loading corpus from {corpus_file}...")
        
        with open(corpus_file, 'r') as f:
            corpus = json.load(f)
        
        print(f"Loaded {len(corpus)} training examples")
        return corpus
    
    def corpus_to_training_data(self, corpus: List[dict]) -> List[Tuple[torch.Tensor, List[str]]]:
        """Convert corpus entries to model training format"""
        training_data = []
        
        print("Converting corpus to training format...")
        
        for entry in corpus:
            try:
                # Extract intent features  
                intent = entry["intent"]
                intent_features = self.model.extract_intent_features(intent)
                
                # Get action sequence (already tokenized)
                action_sequence = entry["action_sequence"]
                
                # Add to training data
                training_data.append((intent_features, action_sequence))
                
            except Exception as e:
                print(f"Error processing {entry['id']}: {e}")
                continue
        
        print(f"Prepared {len(training_data)} training examples")
        return training_data
    
    def train_on_corpus(self, corpus_file: str, num_epochs: int = 20, validation_split: float = 0.2):
        """Train model on corpus with validation split"""
        
        # Load corpus
        corpus = self.load_corpus(corpus_file)
        
        # Convert to training format
        training_data = self.corpus_to_training_data(corpus)
        
        # Split into train/validation
        random.shuffle(training_data)
        split_idx = int(len(training_data) * (1 - validation_split))
        
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        print(f"Training set: {len(train_data)} examples")
        print(f"Validation set: {len(val_data)} examples")
        
        print(f"Training for {num_epochs} epochs...")
        
        self.model.model.train()
        
        for epoch in range(num_epochs):
            epoch_train_loss = 0.0
            
            # Shuffle training data
            random.shuffle(train_data)
            
            # Training loop
            for intent_features, tokens in train_data:
                # Encode tokens
                token_ids = self.model.vocab.encode(tokens)
                
                # Skip if sequence too long
                if len(token_ids) > self.model.config.max_seq_len:
                    continue
                
                # Prepare tensors
                intent_batch = intent_features.unsqueeze(0)  # Add batch dimension
                input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long)  # All but last
                target_ids = torch.tensor([token_ids[1:]], dtype=torch.long)   # All but first
                
                # Forward pass
                logits = self.model.model(intent_batch, input_ids)
                
                # Compute loss
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, self.model.config.vocab_size),
                    target_ids.reshape(-1)
                )
                
                # Backward pass
                self.model.optimizer.zero_grad()
                loss.backward()
                self.model.optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # Validation loop
            self.model.model.eval()
            epoch_val_loss = 0.0
            
            with torch.no_grad():
                for intent_features, tokens in val_data:
                    token_ids = self.model.vocab.encode(tokens)
                    
                    if len(token_ids) > self.model.config.max_seq_len:
                        continue
                    
                    intent_batch = intent_features.unsqueeze(0)
                    input_ids = torch.tensor([token_ids[:-1]], dtype=torch.long)
                    target_ids = torch.tensor([token_ids[1:]], dtype=torch.long)
                    
                    logits = self.model.model(intent_batch, input_ids)
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, self.model.config.vocab_size),
                        target_ids.reshape(-1)
                    )
                    
                    epoch_val_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = epoch_train_loss / len(train_data)
            avg_val_loss = epoch_val_loss / len(val_data) if val_data else 0
            
            print(f"Epoch {epoch+1}/{num_epochs}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")
        
        # Save the trained model
        torch.save({
            'model_state_dict': self.model.model.state_dict(),
            'config': self.model.config.__dict__,
            'epoch': num_epochs,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        }, 'trained_model.pt')
        
        print("✅ Model saved to trained_model.pt")
    
    def evaluate_on_samples(self, corpus_file: str, num_samples: int = 5):
        """Evaluate model on sample corpus entries"""
        corpus = self.load_corpus(corpus_file)
        
        print(f"\n=== Evaluating on {num_samples} samples ===")
        
        samples = random.sample(corpus, min(num_samples, len(corpus)))
        
        for i, entry in enumerate(samples):
            print(f"\nSample {i+1}: {entry['id']}")
            
            intent = entry["intent"]
            true_sequence = entry["action_sequence"]
            
            print(f"  Input: {intent['from_kv']}kV -> {intent['to_kv']}kV, {intent['rating_mva']}MVA")
            
            # Generate from intent
            generated = self.model.generate(intent, temperature=0.5, num_candidates=1)[0]
            
            print(f"  True ({len(true_sequence)} tokens): {' '.join(true_sequence[:10])} ...")
            print(f"  Generated ({len(generated)} tokens): {' '.join(generated[:10])} ...")
            
            # Simple similarity metric
            common_tokens = set(true_sequence) & set(generated)
            jaccard = len(common_tokens) / len(set(true_sequence) | set(generated))
            print(f"  Similarity (Jaccard): {jaccard:.3f}")


def demo_corpus_training():
    """Demonstrate training on realistic corpus"""
    
    # Create model
    config = MinimalConfig()
    model = MinimalActionModel(config)
    trainer = CorpusTrainer(model)
    
    # Train on corpus
    corpus_file = "focused_action_corpus.json"
    trainer.train_on_corpus(corpus_file, num_epochs=30)
    
    # Evaluate
    trainer.evaluate_on_samples(corpus_file)
    
    # Test generation on new intent
    test_intent = {
        "from_kv": 138,
        "to_kv": 34.5,
        "rating_mva": 100,
        "scheme_hv": "double_busbar", 
        "lv_feeders": 4,
        "protection": "standard"
    }
    
    print("\n=== Testing on new intent ===")
    print(f"Intent: {test_intent}")
    
    options = model.generate_design_options(test_intent)
    
    for approach, tokens in options.items():
        print(f"\n{approach.upper()}: {' '.join(tokens[:15])} ...")
    
    return model


if __name__ == "__main__":
    demo_corpus_training()
