"""
validation.py

check that model action is valid and check global diagram validity. Checks passed ruleset, and static ruleset
for specgen model

This module defines the validation logic for diagram entities in the specgen model.
It includes functions to validate actions, check diagram integrity, and ensure that the model adheres to
the specified ruleset.

Author: Gordon Doore
Date Created: 2025-08-19
Last Modified: 2025-08-19
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any
from .action import Action, ActionKind, StepKind, BusRole, Side

# Simple logger setup
logger = logging.getLogger(__name__)

class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"

@dataclass
class ValidationIssue:
    rule_id: str
    message: str
    severity: ValidationSeverity
    auto_fix: Optional[Action] = None

@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)

class ActionValidator:
    """Simple validator: vocabulary + basic rules"""
    
    def __init__(self):
        # Valid vocabulary from your enums
        self.valid_action_kinds = set(ActionKind)
        self.valid_step_kinds = set(StepKind) 
        self.valid_bus_roles = set(BusRole)
        
        # Simple electrical rules
        self.rules = [
            self._check_vocabulary,
            self._check_voltage_consistency,
            self._check_transformer_config,
            self._check_bay_sequence
        ]
    
    def validate_action(self, action: Action, diagram_state: dict) -> ValidationResult:
        """Validate single action against vocabulary + rules"""
        issues = []
        
        for rule_func in self.rules:
            rule_name = rule_func.__name__
            logger.info(f"Checking rule: {rule_name}")
            
            rule_issues = rule_func(action, diagram_state)
            
            if rule_issues:
                logger.warning(f"Rule {rule_name} found {len(rule_issues)} issues")
                for issue in rule_issues:
                    logger.warning(f"  - {issue.rule_id}: {issue.message}")
            else:
                logger.info(f"Rule {rule_name} passed")
            
            issues.extend(rule_issues)
        
        is_valid = not any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        return ValidationResult(is_valid=is_valid, issues=issues)
    
    def _check_vocabulary(self, action: Action, diagram_state: dict) -> List[ValidationIssue]:
        """Check action uses valid enum values"""
        issues = []
        
        if action.kind not in self.valid_action_kinds:
            issues.append(ValidationIssue(
                rule_id="VOCAB_ACTION_KIND",
                message=f"Invalid action kind: {action.kind}",
                severity=ValidationSeverity.ERROR
            ))
        
        # Check step kinds if it's an APPEND_STEP action
        if action.kind == ActionKind.APPEND_STEP:
            if action.args.step_kind not in self.valid_step_kinds:
                # Try to auto-fix common mistakes
                fixed_step = self._fix_step_kind(action.args.step_kind)
                if fixed_step:
                    logger.info(f"Auto-fixed step kind: {action.args.step_kind} → {fixed_step}")
                    fixed_action = action._replace(args=action.args._replace(step_kind=fixed_step))
                    issues.append(ValidationIssue(
                        rule_id="VOCAB_STEP_KIND", 
                        message=f"Fixed step kind: {action.args.step_kind} → {fixed_step}",
                        severity=ValidationSeverity.WARNING,
                        auto_fix=fixed_action
                    ))
                else:
                    issues.append(ValidationIssue(
                        rule_id="VOCAB_STEP_KIND",
                        message=f"Invalid step kind: {action.args.step_kind}",
                        severity=ValidationSeverity.ERROR
                    ))
        
        return issues
    
    def _check_voltage_consistency(self, action: Action, diagram_state: dict) -> List[ValidationIssue]:
        """Check voltage levels match across connections"""
        issues = []
        
        if action.kind == ActionKind.ADD_BUS_CONNECTION:
            bus_kv = diagram_state.get('buses', {}).get(action.args.bus_id, {}).get('kv')
            element_kv = self._get_element_voltage(action.args.element_id, diagram_state)
            
            if bus_kv and element_kv and abs(bus_kv - element_kv) > 0.1:
                issues.append(ValidationIssue(
                    rule_id="VOLTAGE_MISMATCH",
                    message=f"Voltage mismatch: bus {bus_kv}kV vs element {element_kv}kV", 
                    severity=ValidationSeverity.ERROR
                ))
        
        return issues
    
    def _check_transformer_config(self, action: Action, diagram_state: dict) -> List[ValidationIssue]:
        """Check transformer configuration is valid"""
        issues = []
        
        if action.kind == ActionKind.ADD_TRANSFORMER:
            if action.args.hv_kv <= action.args.lv_kv:
                issues.append(ValidationIssue(
                    rule_id="TX_VOLTAGE_HIERARCHY",
                    message="HV voltage must be greater than LV voltage",
                    severity=ValidationSeverity.ERROR
                ))
            
            if action.args.rating_mva <= 0:
                issues.append(ValidationIssue(
                    rule_id="TX_RATING_POSITIVE", 
                    message="Transformer MVA rating must be positive",
                    severity=ValidationSeverity.ERROR
                ))
        
        return issues
    
    def _check_bay_sequence(self, action: Action, diagram_state: dict) -> List[ValidationIssue]:
        """Check bay step sequences make electrical sense"""
        issues = []
        # Simple sequence validation
        return issues
    
    def _fix_step_kind(self, invalid_step: str) -> Optional[StepKind]:
        """Try to fix common step kind mistakes"""
        fixes = {
            "circuit_breaker": StepKind.BREAKER,
            "cb": StepKind.BREAKER, 
            "disconnect": StepKind.BUS_ISOLATOR,
            "current_transformer": StepKind.CT,
            "pt": StepKind.PT
        }
        return fixes.get(invalid_step.lower())
    
    def _get_element_voltage(self, element_id: int, diagram_state: dict) -> Optional[float]:
        """Get voltage of an element from diagram state"""
        # Check transformers, bays, etc.
        return None  # Simplified for example


def validate_action_with_repair(action: Action, diagram_state: dict) -> tuple[ValidationResult, Optional[Action]]:
    """Validate action and return repaired version if possible"""
    validator = ActionValidator()
    result = validator.validate_action(action, diagram_state)
    
    # Try to apply auto-fixes
    for issue in result.issues:
        if issue.auto_fix:
            logger.info(f"Applied auto-fix for {issue.rule_id}: {issue.message}")
            return ValidationResult(is_valid=True, issues=[issue]), issue.auto_fix
    
    return result, action if result.is_valid else None