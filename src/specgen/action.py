"""
action.py

This module defines the action space for the specgen portion of the project
it includes all valid actions that the model can take in generating specfications for 
transformer blueprints

Author: Gordon Doore
Date Created: 2025-08-19
Last Modified: 2025-08-19

"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict,Optional

# ___________________________________________________
# Operation Enums
# ___________________________________________________

class BusRole(str, Enum):
    MAIN = "main"
    SECTION = "section"
    TIE = "tie"

class Side(str, Enum):
    HV = "HV"
    LV = "LV"
    TERT = "TERT"

class Scheme(str, Enum):
    SINGLE_BUSBAR = "single_busbar"
    DOUBLE_BUSBAR = "double_busbar"

class BayKind(str, Enum):
    TRANSFORMER_BAY = "transformer_bay"
    FEEDER_BAY = "feeder_bay"
    LINE_BAY = "line_bay"
    SHUNT_BAY = "shunt_bay"

class StepKind(str, Enum):
    BUS_ISOLATOR = "bus_isolator"
    BREAKER = "breaker"
    LINE_ISOLATOR = "line_isolator"
    CT = "ct"
    PT = "pt"
    ARRESTER = "arrester"
    PLACEHOLDER = "placeholder"

class TransformerKind(str, Enum):
    THREE_WINDING = "three_winding"
    TWO_WINDING = "two_winding"
    AUTOTRANSFORMER = "autotransformer"

class GroundingKind(str, Enum):
    SOLID = "solid"
    NGR = "ngr"  #neutral grounding resistor
    REACTOR = "reactor"


class ShuntKind(str, Enum):
    CAPACITOR = "capacitor"
    # REACTOR = "reactor"


# ___________________________________________________
# Action Space Definition
# ___________________________________________________

class ActionKind(str, Enum):
    """
    Enum-like class to define action types for specgen model
    """
    ADD_BUS = "ADD_BUS"
    ADD_COUPLER = "ADD_COUPLER"
    ADD_TRANSFORMER = "ADD_TRANSFORMER"
    ADD_BAY = "ADD_BAY"
    APPEND_STEP = "APPEND_STEP"
    ADD_BUS_CONNECTION = "ADD_BUS_CONNECTION"
    ADD_SHUNT = "ADD_SHUNT"
    EMIT_SPEC = "EMIT_SPEC"


# ___________________________________________________
# Data Classes for Action Parameters
# ___________________________________________________


@dataclass
class AddBusArgs:
    diagram_id: int
    kv: float
    role: BusRole

@dataclass
class AddCouplerArgs:
    diagram_id: int
    kv: float
    busA_id: int
    busB_id: int


@dataclass
class AddTransformerArgs:
    diagram_id: int
    kind: TransformerKind
    hv_kv: float
    lv_kv: float
    rating_mva: float
    tert_kv: Optional[float] = None  # Optional, only for three-winding transformers
    vector_group: Optional[str] = None
    impedance_percent: Optional[float] = None
    ref_mva: Optional[float] = None
    grounding_hv: Optional[GroundingKind] = None
    grounding_lv: Optional[GroundingKind] = None
    grounding_tert: Optional[GroundingKind] = None

@dataclass
class AddBayArgs:
    diagram_id: int
    side: Side
    kv: float
    kind: BayKind
    scheme: Optional[Scheme] = None
    future: bool = False # Indicates if the bay is for future use

@dataclass
class AppendStepArgs:
    bay_id: int
    step_kind: StepKind

@dataclass
class AddBusConnectionArgs:
    diagram_id: int
    bus_id: int
    element_id: int  # ID of transformer, bay, or shunt being connected
    element_side: Optional[Side] = None  # Side of the element to connect to the bus

@dataclass
class AddShuntArgs:
    diagram_id: int
    kind: ShuntKind
    kv: float
    mvar: float
    bus_id: int
    # steps_mvar: Optional[List[Tuple[StepKind, float]]] = None # List of (step_kind, mvar) tuples for shunt steps
    banks: Optional[int] = None  # Number of banks for the shunt, if applicable
    

@dataclass
class EmitSpecArgs:
    pass

# ___________________________________________________
# Action carrier + registry
# ___________________________________________________

@dataclass
class Action:
    kind: ActionKind
    args: Any #maps to *Args of dataclass types defined above

    def __post_init__(self):
        validate_action(self)

ACTION_ARG_SCHEMA: Dict[ActionKind, Any] = {
    ActionKind.ADD_BUS: AddBusArgs,
    ActionKind.ADD_COUPLER: AddCouplerArgs,
    ActionKind.ADD_TRANSFORMER: AddTransformerArgs,
    ActionKind.ADD_BAY: AddBayArgs,
    ActionKind.APPEND_STEP: AppendStepArgs,
    ActionKind.ADD_BUS_CONNECTION: AddBusConnectionArgs,
    ActionKind.ADD_SHUNT: AddShuntArgs,
    ActionKind.EMIT_SPEC: EmitSpecArgs,
}



def validate_action(action: Action):
    """
    Validate the action kind and its arguments against the defined schema.
    Raises ValueError if validation fails.
    """
    if action.kind not in ACTION_ARG_SCHEMA:
        raise ValueError(f"Invalid action kind: {action.kind}")

    expected_args_type = ACTION_ARG_SCHEMA[action.kind]
    if not isinstance(action.args, expected_args_type):
        raise ValueError(f"Invalid args type for {action.kind}: expected {expected_args_type}, got {type(action.args)}")


