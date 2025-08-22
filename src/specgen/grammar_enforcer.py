"""
grammar_enforcer.py

Defines the GrammarEnforcer class that validates action sequences using
the DSL grammar and parser. It converts model token output to DSL format
and validates using the Lark parser and validator.

Author: Gordon Doore
Date Created: 2025-08-22
Last Modified: 2025-08-22
"""

import torch
from typing import List, Set, Tuple, Optional
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.specgen.vocabulary import ActionVocabulary
from src.dsl.parser import parse, ParseError
from src.dsl.validator import validate, DSLValidationError


class GrammarEnforcer:
    """Enforces DSL grammar rules for action token sequences"""
    
    def __init__(self, vocab: ActionVocabulary):
        self.vocab = vocab
        self.token_sets = vocab.get_token_sets()
        
        # Build state machine for valid next tokens based on DSL grammar
        self._build_grammar_state_machine()
    
    def _build_grammar_state_machine(self):
        """Build state machine for valid token transitions based on DSL grammar"""
        
        # States in our grammar parsing - simplified to match actual DSL grammar
        self.STATES = {
            'START': 'START',
            'EXPECT_ACTION': 'EXPECT_ACTION',
            'EXPECT_PARAMS': 'EXPECT_PARAMS',
            'EXPECT_PARAM_VALUE': 'EXPECT_PARAM_VALUE',
            'INSIDE_LIST': 'INSIDE_LIST',
            'ACTION_COMPLETE': 'ACTION_COMPLETE',
            'COMPLETE': 'COMPLETE'
        }
        
        # Parameter sequences for each action type (based on DSL grammar)
        self.action_param_sequences = {
            'ADD_BUS': ['id=', 'ID', 'kv=', 'VOLTAGE'],
            'ADD_BAY': ['id=', 'ID', 'kind=', 'BAY_KIND', 'kv=', 'VOLTAGE', 'bus=', 'ID'],
            'ADD_COUPLER': ['id=', 'ID', 'kv=', 'VOLTAGE', 'from_bus=', 'ID', 'to_bus=', 'ID'],
            'ADD_BREAKER': ['id=', 'ID', 'kv=', 'VOLTAGE', 'interrupting_kA=', 'NUMERIC', 'kind=', 'BREAKER_KIND', 'continuous_A=', 'NUMERIC'],
            'ADD_DISCONNECTOR': ['id=', 'ID', 'kv=', 'VOLTAGE', 'kind=', 'DISCONNECTOR_KIND', 'continuous_A=', 'NUMERIC'],
            'ADD_TRANSFORMER': ['id=', 'ID', 'kind=', 'TRANSFORMER_KIND', 'kv_in=', 'VOLTAGE', 'kv_out=', 'VOLTAGE', 'tert_kv=', 'VOLTAGE', 'rated_MVA=', 'MVA', 'vector_group=', 'STRING', 'percentZ=', 'NUMERIC'],
            'ADD_LINE': ['id=', 'ID', 'kv=', 'VOLTAGE', 'kind=', 'LINE_KIND', 'length_km=', 'NUMERIC', 'thermal_A=', 'NUMERIC'],
            'CONNECT': ['series=', 'LIST_START'],
            'APPEND_TO_BAY': ['bay_id=', 'ID', 'object_id=', 'ID'],
            'VALIDATE': [],
            'EMIT_SPEC': []
        }
    
    def get_valid_next_tokens(self, sequence: List[str], current_state: str = None) -> Set[str]:
        """Get set of valid next tokens given current sequence"""
        
        if not sequence:
            return {'<START>'}
        
        if sequence[-1] == '<START>':
            return self.token_sets['actions']
        
        # Check if we just completed EMIT_SPEC - only <END> is allowed
        if sequence and sequence[-1] == 'EMIT_SPEC':
            return {'<END>'}
        
        # Special handling for inside list/bracket context (for CONNECT statements)
        bracket_count = 0
        for token in sequence:
            if token == '[':
                bracket_count += 1
            elif token == ']':
                bracket_count -= 1
        
        # If we're inside brackets, only allow ID tokens, commas, and closing bracket
        if bracket_count > 0:
            # Find the last bracket opening to understand context
            last_bracket_idx = -1
            for i in range(len(sequence) - 1, -1, -1):
                if sequence[i] == '[':
                    last_bracket_idx = i
                    break
            
            # Get tokens since bracket opening
            if last_bracket_idx >= 0:
                tokens_in_bracket = sequence[last_bracket_idx + 1:]
                
                # If last token was a comma or we just opened bracket, allow ID tokens
                if not tokens_in_bracket or tokens_in_bracket[-1] == ',':
                    return (self.token_sets['bus_ids'] | 
                           self.token_sets['breaker_ids'] |
                           self.token_sets['coupler_ids'] |
                           self.token_sets['bay_ids'] |
                           self.token_sets['disconnector_ids'] |
                           self.token_sets['transformer_ids'] |
                           self.token_sets['line_ids'])
                
                # If last token was an ID, allow comma or closing bracket
                last_token = tokens_in_bracket[-1]
                if any(last_token in self.token_sets.get(id_set, set()) 
                       for id_set in ['bus_ids', 'breaker_ids', 'coupler_ids', 'bay_ids', 
                                     'disconnector_ids', 'transformer_ids', 'line_ids']):
                    return {',', ']'}
                
                # If we have comma + ID + comma pattern, allow more IDs or closing
                if len(tokens_in_bracket) >= 2:
                    return {']'}  # Can always close bracket
            
            # Fallback: allow closing bracket
            return {']'}
        
        # Find current action context and validate parameter sequence
        current_action = None
        action_start_idx = -1
        
        # Scan backwards to find the most recent action
        for i in range(len(sequence) - 1, -1, -1):
            if sequence[i] in self.token_sets['actions']:
                current_action = sequence[i]
                action_start_idx = i
                break
        
        if not current_action:
            # No action found, can start new action or emit spec to end
            return self.token_sets['actions'] | {'EMIT_SPEC'}
        
        # Check if current action has a parameter sequence to follow
        if current_action in self.action_param_sequences:
            expected_params = self.action_param_sequences[current_action]
            tokens_since_action = sequence[action_start_idx + 1:]
            
            # Validate that the sequence so far matches the expected pattern
            validation_error = self._validate_parameter_sequence(tokens_since_action, expected_params)
            if validation_error:
                # Sequence is invalid, return empty set to force error
                return set()
            
            if len(tokens_since_action) >= len(expected_params):
                # For CONNECT, after completing basic params, we're in list context
                if current_action == 'CONNECT' and '[' in tokens_since_action and ']' not in tokens_since_action:
                    # We're inside the list, this should have been handled above
                    return {']'}  # Force bracket closure
                
                # Action is complete, can start new action or emit spec to end sequence
                return self.token_sets['actions'] | {'EMIT_SPEC'}
            
            # Determine exactly what parameter type is expected next
            param_idx = len(tokens_since_action)
            expected_param_type = expected_params[param_idx]
            
            # Return only the specific tokens valid for this parameter position
            return self._get_tokens_for_param_type(expected_param_type, sequence)
        
        # Handle special actions that don't have parameters
        if current_action in ['VALIDATE', 'EMIT_SPEC']:
            if current_action == 'EMIT_SPEC':
                return {'<END>'}
            else:  # VALIDATE
                return self.token_sets['actions'] | {'EMIT_SPEC'}
        
        # Default fallback - can start new actions or emit spec
        return self.token_sets['actions'] | {'EMIT_SPEC'}
    
    def _get_tokens_for_param_type(self, param_type: str, sequence: List[str] = None) -> Set[str]:
        """Get valid tokens for a specific parameter type"""
        
        if param_type in self.token_sets['params']:
            return {param_type}
        elif param_type == 'ID':
            # Return all ID types
            return (self.token_sets['bus_ids'] | 
                   self.token_sets['breaker_ids'] |
                   self.token_sets['coupler_ids'] |
                   self.token_sets['bay_ids'] |
                   self.token_sets['disconnector_ids'] |
                   self.token_sets['transformer_ids'] |
                   self.token_sets['line_ids'])
        elif param_type == 'VOLTAGE':
            return self.token_sets['voltages']
        elif param_type == 'MVA':
            return self.token_sets['mva_values']
        elif param_type == 'BAY_KIND':
            return self.token_sets['bay_kinds']
        elif param_type == 'BREAKER_KIND':
            return self.token_sets['breaker_kinds']
        elif param_type == 'DISCONNECTOR_KIND':
            return self.token_sets['disconnector_kinds']
        elif param_type == 'TRANSFORMER_KIND':
            return self.token_sets['transformer_kinds']
        elif param_type == 'LINE_KIND':
            return self.token_sets['line_kinds']
        elif param_type == 'STRING':
            # For strings like vector_group, we'll allow some predefined values
            return {'Dd0', 'Yy0', 'Dy11', 'Yz11'}  # Common transformer vector groups
        elif param_type == 'NUMERIC':
            # For numeric values, check context to determine what specific values are valid
            numeric_tokens = {'1000', '2000', '3000', '5000', '10000', '25000'}
            
            # If we have sequence context, check for specific parameter types
            if sequence and len(sequence) >= 2:
                prev_token = sequence[-1]  # This should be the parameter name like 'percentZ='
                
                if prev_token == 'percentZ=':
                    # After percentZ=, allow percentZ values
                    percent_z_tokens = {token for token in self.vocab.vocab if token.endswith('percentZ')}
                    return percent_z_tokens
                elif prev_token == 'interrupting_kA=':
                    # After interrupting_kA=, allow kA values
                    ka_tokens = {token for token in self.vocab.vocab if token.endswith('kA')}
                    return ka_tokens
                elif prev_token == 'continuous_A=':
                    # After continuous_A=, allow A values
                    a_tokens = {token for token in self.vocab.vocab if token.endswith('A') and not token.endswith('kA') and not token.endswith('MVA')}
                    return a_tokens
                
            return numeric_tokens
        elif param_type == 'LIST_START':
            return {'['}
        else:
            return {param_type} if param_type in self.vocab.vocab else set()
    
    def mask_invalid_tokens(self, logits: torch.Tensor, sequence: List[str]) -> torch.Tensor:
        """Apply grammar mask to logits, setting invalid tokens to -inf"""
        valid_tokens = self.get_valid_next_tokens(sequence)
        valid_ids = []
        
        for token in valid_tokens:
            if token in self.vocab.token_to_id:
                valid_ids.append(self.vocab.token_to_id[token])
            else:
                # Handle unknown tokens by finding closest match
                if token.endswith('kv'):
                    # Map to closest voltage
                    for vocab_token in self.token_sets['voltages']:
                        if vocab_token in self.vocab.token_to_id:
                            valid_ids.append(self.vocab.token_to_id[vocab_token])
                            break
        
        # Create mask - start with all tokens masked
        mask = torch.full_like(logits, float('-inf'))
        
        # Unmask valid tokens
        if valid_ids:
            mask[valid_ids] = 0
        else:
            # Fallback: allow UNK token
            if '<UNK>' in self.vocab.token_to_id:
                mask[self.vocab.token_to_id['<UNK>']] = 0
        
        return logits + mask
    
    def tokens_to_dsl(self, tokens: List[str]) -> str:
        """Convert model token sequence to DSL text format"""
        
        # Filter out special tokens
        filtered_tokens = [t for t in tokens if t not in {'<START>', '<END>', '<PAD>', '<UNK>'}]
        
        if not filtered_tokens:
            return ""
        
        dsl_lines = []
        i = 0
        
        while i < len(filtered_tokens):
            token = filtered_tokens[i]
            
            # Handle actions
            if token in self.token_sets['actions']:
                
                if token.startswith('ADD_'):
                    # Handle ADD_* actions with proper grammar formatting
                    component_type = token[4:]  # Remove 'ADD_'
                    action_name = f"ADD_{component_type}"
                    i += 1
                    
                    # Collect parameters as key=value pairs
                    param_pairs = []
                    while i < len(filtered_tokens) and filtered_tokens[i] not in self.token_sets['actions']:
                        param_token = filtered_tokens[i]
                        
                        if param_token.endswith('='):
                            # Parameter name - get the value
                            if i + 1 < len(filtered_tokens):
                                value_token = filtered_tokens[i + 1]
                                # Convert value token to appropriate format
                                converted_value = self._convert_value_token(value_token)
                                param_pairs.append(f"{param_token}{converted_value}")
                                i += 2
                            else:
                                i += 1
                        else:
                            i += 1
                    
                    # Format according to grammar: "ADD_BUS id=bus100, kv=230"
                    if param_pairs:
                        dsl_lines.append(f"{action_name} {', '.join(param_pairs)}")
                    else:
                        dsl_lines.append(action_name)
                
                elif token == 'CONNECT':
                    i += 1
                    # Handle series list
                    series_items = []
                    
                    while i < len(filtered_tokens) and filtered_tokens[i] not in self.token_sets['actions']:
                        if filtered_tokens[i] == 'series=':
                            i += 1
                        elif filtered_tokens[i] == '[':
                            i += 1  # skip opening bracket
                        elif filtered_tokens[i] == ']':
                            i += 1  # skip closing bracket
                            break
                        elif filtered_tokens[i] == ',':
                            i += 1  # skip comma
                        else:
                            series_items.append(filtered_tokens[i])
                            i += 1
                    
                    # Format according to grammar: "CONNECT series=[item1, item2]"
                    if series_items:
                        dsl_lines.append(f"CONNECT series=[{', '.join(series_items)}]")
                    else:
                        dsl_lines.append("CONNECT series=[]")
                
                elif token == 'APPEND_TO_BAY':
                    i += 1
                    # Collect parameters
                    param_pairs = []
                    while i < len(filtered_tokens) and filtered_tokens[i] not in self.token_sets['actions']:
                        param_token = filtered_tokens[i]
                        
                        if param_token.endswith('='):
                            if i + 1 < len(filtered_tokens):
                                value_token = filtered_tokens[i + 1]
                                converted_value = self._convert_value_token(value_token)
                                param_pairs.append(f"{param_token}{converted_value}")
                                i += 2
                            else:
                                i += 1
                        else:
                            i += 1
                    
                    # Format: "APPEND_TO_BAY bay_id=bay1, object_id=obj1"
                    if param_pairs:
                        dsl_lines.append(f"APPEND_TO_BAY {', '.join(param_pairs)}")
                    else:
                        dsl_lines.append("APPEND_TO_BAY")
                
                elif token in ['VALIDATE', 'EMIT_SPEC']:
                    dsl_lines.append(token)
                    i += 1
                    
                    # Skip any parameters (these commands typically don't have params)
                    while i < len(filtered_tokens) and filtered_tokens[i] not in self.token_sets['actions']:
                        i += 1
            else:
                i += 1
        
        return '\n'.join(dsl_lines)
    
    def _convert_value_token(self, token: str) -> str:
        """Convert vocabulary token to DSL format"""
        
        # Handle voltage values: "230kv" -> "230"
        if token.endswith('kv'):
            return token[:-2]  # Remove 'kv' suffix
        
        # Handle MVA values: "50mva" -> "50"  
        if token.endswith('mva'):
            return token[:-3]  # Remove 'mva' suffix
        
        # Handle percent impedance: "8percentZ" -> "8"
        if token.endswith('percentZ'):
            return token[:-8]  # Remove 'percentZ' suffix
        
        # Handle kA values: "31.5kA" -> "31.5" (must come before 'A' check)
        if token.endswith('kA'):
            return token[:-2]  # Remove 'kA' suffix
        
        # Handle amperage values: "1200A" -> "1200"
        if token.endswith('A') and token not in {'Dd0', 'Yy0', 'Dy11', 'Yz11'}:
            return token[:-1]  # Remove 'A' suffix
        
        # Handle quoted strings - add quotes if not present
        if token in {'Dd0', 'Yy0', 'Dy11', 'Yz11', 'YN0', 'YNd1', 'YNd5', 'YNd11', 'YNd7', 'Yzn5', 'Yzn11', 'Yzn1', 'Yzn7', 'Yy6', 'Yy12', 'Dd6', 'Dd12', 'Dyn1', 'Dyn5', 'Dyn11', 'Dyn7', 'Dzn5', 'Dzn11', 'Dzn1', 'Dzn7'}:
            return f'"{token}"'
        
        # Handle numeric values
        if token.isdigit():
            return token
        
        # Default: return as-is
        return token
    
    def validate_sequence(self, tokens: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate complete token sequence using DSL parser and validator"""
        
        try:
            # Convert tokens to DSL text
            dsl_text = self.tokens_to_dsl(tokens)
            
            if not dsl_text.strip():
                return False, "Empty DSL text"
            
            # Parse using DSL parser
            ir = parse(dsl_text)
            
            # Validate using DSL validator
            validate(ir)
            
            return True, None
            
        except ParseError as e:
            return False, f"Parse error: {e}"
        except DSLValidationError as e:
            return False, f"Validation error: {e}"
        except Exception as e:
            return False, f"Unexpected error: {e}"
    
    def is_sequence_complete(self, tokens: List[str]) -> bool:
        """Check if token sequence represents a complete, valid DSL program"""
        
        # Must have start and end tokens
        if not tokens or tokens[0] != '<START>' or tokens[-1] != '<END>':
            return False
        
        # Must have at least one action
        actions_found = any(token in self.token_sets['actions'] for token in tokens)
        if not actions_found:
            return False
        
        # Must end with EMIT_SPEC before <END>
        if len(tokens) < 3 or tokens[-2] != 'EMIT_SPEC':
            return False
        
        # Validate using DSL parser
        is_valid, _ = self.validate_sequence(tokens)
        return is_valid
    
    def repair_sequence(self, tokens: List[str]) -> List[str]:
        """Attempt to repair an invalid token sequence"""
        
        if not tokens:
            return ['<START>', 'EMIT_SPEC', '<END>']
        
        repaired = []
        
        # Ensure starts with <START>
        if tokens[0] != '<START>':
            repaired.append('<START>')
        
        # Copy valid tokens
        for token in tokens:
            if token == '<START>' and repaired and repaired[0] == '<START>':
                continue  # Skip duplicate START
            if token in self.vocab.vocab:
                repaired.append(token)
        
        # Ensure has at least one action
        has_action = any(token in self.token_sets['actions'] for token in repaired)
        if not has_action:
            # Add a minimal action if none exists
            if len(repaired) == 1:  # Only has <START>
                repaired.append('EMIT_SPEC')
            else:
                # Insert EMIT_SPEC before any existing <END>
                if repaired[-1] == '<END>':
                    repaired.insert(-1, 'EMIT_SPEC')
                else:
                    repaired.append('EMIT_SPEC')
        else:
            # Has actions, ensure EMIT_SPEC comes before <END>
            if '<END>' in repaired:
                end_idx = repaired.index('<END>')
                if end_idx > 0 and repaired[end_idx - 1] != 'EMIT_SPEC':
                    # Insert EMIT_SPEC before <END>
                    repaired.insert(end_idx, 'EMIT_SPEC')
            else:
                # No <END> yet, add EMIT_SPEC
                repaired.append('EMIT_SPEC')
        
        # Ensure ends with <END>
        if not repaired or repaired[-1] != '<END>':
            repaired.append('<END>')
        
        return repaired
    
    def get_completion_suggestions(self, partial_tokens: List[str]) -> List[str]:
        """Get suggested completions for a partial token sequence"""
        
        valid_next = self.get_valid_next_tokens(partial_tokens)
        
        # Sort suggestions by frequency/preference
        suggestions = list(valid_next)
        
        # Prioritize common actions and end token
        priority_tokens = ['<END>', 'ADD_BUS', 'ADD_TRANSFORMER', 'CONNECT', 'EMIT_SPEC']
        priority_suggestions = [t for t in priority_tokens if t in suggestions]
        other_suggestions = [t for t in suggestions if t not in priority_tokens]
        
        return priority_suggestions + sorted(other_suggestions)
    
    def _validate_parameter_sequence(self, tokens_so_far: List[str], expected_params: List[str]) -> Optional[str]:
        """
        Validate that the parameter sequence so far matches the expected pattern.
        Returns error message if invalid, None if valid.
        """
        
        if len(tokens_so_far) > len(expected_params):
            return f"Too many parameters: got {len(tokens_so_far)}, expected at most {len(expected_params)}"
        
        # Check each token against the expected parameter pattern
        for i, token in enumerate(tokens_so_far):
            expected_type = expected_params[i]
            
            # Check if the token is valid for this position
            if not self._is_token_valid_for_type(token, expected_type):
                return f"Invalid token '{token}' at position {i}, expected type '{expected_type}'"
        
        return None  # All tokens are valid so far
    
    def _is_token_valid_for_type(self, token: str, expected_type: str) -> bool:
        """Check if a specific token is valid for the expected parameter type"""
        
        # Handle parameter names (like 'id=', 'kv=')
        if expected_type.endswith('='):
            return token == expected_type
        
        # Handle value types
        elif expected_type == 'ID':
            return (token in self.token_sets['bus_ids'] or 
                   token in self.token_sets['breaker_ids'] or
                   token in self.token_sets['coupler_ids'] or
                   token in self.token_sets['bay_ids'] or
                   token in self.token_sets['disconnector_ids'] or
                   token in self.token_sets['transformer_ids'] or
                   token in self.token_sets['line_ids'])
        
        elif expected_type == 'VOLTAGE':
            return token in self.token_sets['voltages']
        
        elif expected_type == 'MVA':
            return token in self.token_sets['mva_values']
        
        elif expected_type == 'BAY_KIND':
            return token in self.token_sets['bay_kinds']
        
        elif expected_type == 'BREAKER_KIND':
            return token in self.token_sets['breaker_kinds']
        
        elif expected_type == 'DISCONNECTOR_KIND':
            return token in self.token_sets['disconnector_kinds']
        
        elif expected_type == 'TRANSFORMER_KIND':
            return token in self.token_sets['transformer_kinds']
        
        elif expected_type == 'LINE_KIND':
            return token in self.token_sets['line_kinds']
        
        elif expected_type == 'STRING':
            # For transformer vector groups and other strings
            return token in {'Dd0', 'Yy0', 'Dy11', 'Yz11'} or token.startswith('"')
        
        elif expected_type == 'NUMERIC':
            # Allow numeric tokens or predefined numeric values
            if (token.isdigit() or 
                token in {'1000', '2000', '3000', '5000', '10000', '25000'} or
                token.replace('.', '').replace('-', '').isdigit()):
                return True
                
            # Allow tokens with unit suffixes (these are treated as numeric values)
            if (token.endswith('percentZ') or 
                token.endswith('kA') or 
                token.endswith('mva') or 
                token.endswith('kv') or
                (token.endswith('A') and not token.endswith('kA') and not token.endswith('MVA'))):
                # Check if the part before the unit is numeric
                if token.endswith('percentZ'):
                    numeric_part = token[:-8]  # Remove 'percentZ'
                elif token.endswith('kA'):
                    numeric_part = token[:-2]   # Remove 'kA' 
                elif token.endswith('mva'):
                    numeric_part = token[:-3]   # Remove 'mva'
                elif token.endswith('kv'):
                    numeric_part = token[:-2]   # Remove 'kv'
                elif token.endswith('A'):
                    numeric_part = token[:-1]   # Remove 'A'
                else:
                    return False
                
                # Check if numeric part is valid
                return (numeric_part.replace('.', '').replace('-', '').isdigit() and 
                        len(numeric_part) > 0)
            
            return False
        
        elif expected_type == 'LIST_START':
            return token == '['
        
        else:
            # Direct token match
            return token == expected_type

    def check_grammar_violation(self, sequence: List[str]) -> Optional[str]:
        """
        Check if the current sequence violates grammar rules.
        Returns error message if violation found, None if valid.
        """
        
        if not sequence:
            return None
        
        # Find current action context
        current_action = None
        action_start_idx = -1
        
        for i in range(len(sequence) - 1, -1, -1):
            if sequence[i] in self.token_sets['actions']:
                current_action = sequence[i]
                action_start_idx = i
                break
        
        if current_action and current_action in self.action_param_sequences:
            expected_params = self.action_param_sequences[current_action]
            tokens_since_action = sequence[action_start_idx + 1:]
            
            return self._validate_parameter_sequence(tokens_since_action, expected_params)
        
        return None
    
    def get_current_state(self, sequence: List[str]) -> str:
        """Get the current grammar state based on actual DSL grammar structure"""
        
        if not sequence:
            return 'START'
        
        if sequence[-1] == '<START>':
            return 'EXPECT_ACTION'
        
        if sequence[-1] == 'EMIT_SPEC':
            return 'COMPLETE'
        
        # Check if we're inside a list (for CONNECT statements)
        bracket_count = 0
        for token in sequence:
            if token == '[':
                bracket_count += 1
            elif token == ']':
                bracket_count -= 1
        
        if bracket_count > 0:
            return 'INSIDE_LIST'
        
        # Find current action context
        current_action = None
        action_start_idx = -1
        
        for i in range(len(sequence) - 1, -1, -1):
            if sequence[i] in self.token_sets['actions']:
                current_action = sequence[i]
                action_start_idx = i
                break
        
        if not current_action:
            return 'EXPECT_ACTION'
        
        # Determine state based on current action and parameter progress
        if current_action in self.action_param_sequences:
            expected_params = self.action_param_sequences[current_action]
            tokens_since_action = sequence[action_start_idx + 1:]
            param_idx = len(tokens_since_action)
            
            if param_idx >= len(expected_params):
                # Action complete, expect next action
                return 'ACTION_COMPLETE'
            else:
                # Check if we're expecting a parameter value (after seeing param=)
                if param_idx > 0 and tokens_since_action[-1].endswith('='):
                    return 'EXPECT_PARAM_VALUE'
                else:
                    # We're expecting parameters for this action
                    return 'EXPECT_PARAMS'
        
        # Special handling for actions without parameters
        if current_action in ['VALIDATE', 'EMIT_SPEC']:
            return 'ACTION_COMPLETE'  # These actions complete immediately
        
        return 'EXPECT_ACTION'
    
    def get_expected_next(self, sequence: List[str]) -> Tuple[str, Set[str], Optional[str]]:
        """
        Get detailed information about what's expected next.
        Returns: (current_state, valid_tokens, error_if_any)
        """
        
        current_state = self.get_current_state(sequence)
        error = self.check_grammar_violation(sequence)
        
        if error:
            return current_state, set(), error
        
        valid_tokens = self.get_valid_next_tokens(sequence)
        return current_state, valid_tokens, None
    
    def format_validation_error(self, sequence: List[str]) -> str:
        """Format a detailed validation error message"""
        
        error = self.check_grammar_violation(sequence)
        if not error:
            return "No validation errors found"
        
        state = self.get_current_state(sequence)
        
        # Find current action context for more detail
        current_action = None
        action_start_idx = -1
        
        for i in range(len(sequence) - 1, -1, -1):
            if sequence[i] in self.token_sets['actions']:
                current_action = sequence[i]
                action_start_idx = i
                break
        
        if current_action and current_action in self.action_param_sequences:
            expected_params = self.action_param_sequences[current_action]
            tokens_since_action = sequence[action_start_idx + 1:]
            param_idx = len(tokens_since_action)
            
            if param_idx < len(expected_params):
                expected_next = expected_params[param_idx]
                return (f"Grammar Error in {current_action}:\n"
                       f"  Error: {error}\n"
                       f"  State: {state}\n"
                       f"  Expected next: {expected_next}\n"
                       f"  Parameter sequence: {expected_params}\n"
                       f"  Progress: {param_idx}/{len(expected_params)}")
        
        return f"Grammar Error: {error}\nState: {state}"
    
    def process_sequence_for_training(self, tokens: List[str], repair_if_invalid: bool = True) -> Tuple[List[int], bool, Optional[str]]:
        """
        Unified method to check, repair, and convert a token sequence to indices for training.
        
        Args:
            tokens: Token sequence to process
            repair_if_invalid: Whether to attempt repair if sequence is invalid
            
        Returns:
            Tuple of (token_ids, is_valid, error_message)
            - token_ids: List of token indices ready for training
            - is_valid: Whether the final sequence is grammatically valid
            - error_message: Error description if any, None if valid
        """
        
        # Step 1: Check if sequence is valid
        is_valid, error_message = self.validate_sequence(tokens)
        
        processed_tokens = tokens.copy()
        
        # Step 2: Attempt repair if invalid and repair is enabled
        if not is_valid and repair_if_invalid:
            original_error = error_message
            processed_tokens = self.repair_sequence(tokens)
            
            # Re-validate repaired sequence
            is_valid, error_message = self.validate_sequence(processed_tokens)
            
            if is_valid:
                error_message = f"Original error: {original_error} - Repaired successfully"
            else:
                error_message = f"Original error: {original_error} - Repair failed: {error_message}"
        
        # Step 3: Convert tokens to indices
        token_ids = []
        for token in processed_tokens:
            if token in self.vocab.token_to_id:
                token_ids.append(self.vocab.token_to_id[token])
            else:
                # Handle unknown tokens by mapping to <UNK>
                token_ids.append(self.vocab.token_to_id.get('<UNK>', 0))
        
        return token_ids, is_valid, error_message
    
    def process_and_mask_for_generation(self, sequence: List[str], logits: torch.Tensor) -> Tuple[torch.Tensor, Set[str], str]:
        """
        Unified method for generation: check current sequence and mask invalid next tokens.
        
        Args:
            sequence: Current token sequence being generated
            logits: Model logits for next token prediction
            
        Returns:
            Tuple of (masked_logits, valid_tokens, current_state)
            - masked_logits: Logits with invalid tokens masked to -inf
            - valid_tokens: Set of valid next tokens
            - current_state: Current grammar state for debugging
        """
        
        # Get current state and valid tokens
        current_state, valid_tokens, error = self.get_expected_next(sequence)
        
        # Apply mask to logits
        masked_logits = self.mask_invalid_tokens(logits, sequence)
        
        return masked_logits, valid_tokens, current_state
    
    def batch_process_sequences_for_training(self, token_sequences: List[List[str]], 
                                           repair_invalid: bool = True) -> Tuple[List[List[int]], List[bool], List[str]]:
        """
        Process multiple sequences in batch for training.
        
        Args:
            token_sequences: List of token sequences to process
            repair_invalid: Whether to repair invalid sequences
            
        Returns:
            Tuple of (batch_token_ids, validity_flags, error_messages)
        """
        
        batch_token_ids = []
        validity_flags = []
        error_messages = []
        
        for tokens in token_sequences:
            token_ids, is_valid, error_msg = self.process_sequence_for_training(tokens, repair_invalid)
            
            batch_token_ids.append(token_ids)
            validity_flags.append(is_valid)
            error_messages.append(error_msg or "Valid")
        
        return batch_token_ids, validity_flags, error_messages

# Test function
def test_grammar_enforcer():
    """Test the grammar enforcer with strict state enforcement"""
    
    vocab = ActionVocabulary()
    enforcer = GrammarEnforcer(vocab)
    
    print("Testing Strict Grammar Enforcer:")
    print("=" * 40)
    
    # Test valid sequence
    valid_sequence = ['<START>', 'ADD_BUS', 'id=', 'bus100', 'kv=', '230kv', 'EMIT_SPEC', '<END>']
    
    print("VALID SEQUENCE:")
    for i in range(1, len(valid_sequence) + 1):
        partial = valid_sequence[:i]
        state, valid_next, error = enforcer.get_expected_next(partial)
        
        if error:
            print(f"  Step {i}: {partial[-1]} -> ERROR: {error}")
        else:
            next_tokens = sorted(list(valid_next))[:3]
            print(f"  Step {i}: {partial[-1]} -> {next_tokens} (state: {state})")
    
    print("\nFinal DSL:")
    dsl_text = enforcer.tokens_to_dsl(valid_sequence)
    print(dsl_text)
    
    print("\nValidation result:")
    is_valid, validation_error = enforcer.validate_sequence(valid_sequence)
    print(f"Valid: {is_valid}")
    if validation_error:
        print(f"Error: {validation_error}")
    
    # Test invalid sequence
    print("\nINVALID SEQUENCE:")
    invalid_sequence = ['<START>', 'ADD_BUS', 'kv=', 'bus100']  # Wrong order
    
    for i in range(1, len(invalid_sequence) + 1):
        partial = invalid_sequence[:i]
        state, valid_next, error = enforcer.get_expected_next(partial)
        
        if error:
            print(f"  Step {i}: {partial[-1]} -> ERROR")
            print(f"    {enforcer.format_validation_error(partial)}")
            break
        else:
            next_tokens = sorted(list(valid_next))[:3]
            print(f"  Step {i}: {partial[-1]} -> {next_tokens} (state: {state})")


if __name__ == "__main__":
    test_grammar_enforcer()

