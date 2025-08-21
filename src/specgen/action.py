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
from typing import Any, Dict,Optional, List

# ___________________________________________________
# Operation Enums
# ___________________________________________________


class BayKind(str, Enum):
    TRANSFORMER = "TRANSFORMER"
    FEEDER = "FEEDER"
    LINE = "LINE"
    SHUNT = "SHUNT"
    COUPLER = "COUPLER"
    GENERATOR = "GENERATOR"

class BreakerKind(str, Enum):
    SF6 = "SF6"
    VACUUM = "VACUUM"
    OIL = "OIL"
    AIRBLAST = "AIRBLAST"

class DisconnectorKind(str, Enum):
    CENTER_BREAK = "CENTER_BREAK"
    DOUBLE_BREAK = "DOUBLE_BREAK"
    PANTOGRAPH = "PANTOGRAPH"
    EARTH_SWITCH_COMBINED = "EARTH_SWITCH_COMBINED"

class TransformerKind(str, Enum):
    TWO_WINDING = "TWO_WINDING"
    AUTO = "AUTO"
    THREE_WINDING = "THREE_WINDING"
    GROUNDING = "GROUNDING"

class LineKind(str, Enum):
    OHL = "OHL"  # Overhead Line
    UGC = "UGC"  # Underground Cable


# ___________________________________________________
# Action Space Definition
# ___________________________________________________

class ActionKind(str, Enum):
    """
    Enum-like class to define action types for specgen model
    """
    ADD_BUS = "ADD_BUS"
    ADD_BAY = "ADD_BAY"
    ADD_COUPLER = "ADD_COUPLER"
    ADD_BREAKER = "ADD_BREAKER"
    ADD_DISCONNECTOR = "ADD_DISCONNECTOR"
    ADD_TRANSFORMER = "ADD_TRANSFORMER"
    ADD_LINE = "ADD_LINE"
    CONNECT = "CONNECT"
    APPEND_TO_BAY = "APPEND_TO_BAY"
    VALIDATE = "VALIDATE"
    EMIT_SPEC = "EMIT_SPEC"


# ___________________________________________________
# Data Classes for Action Parameters
# ___________________________________________________


@dataclass
class AddBusArgs:
    diagram_id : str
    kv: float

@dataclass
class AddBreakerArgs:
    diagram_id: str
    kv: float
    interrupting_kv: float
    kind: BreakerKind
    continuous_A: float

@dataclass
class AddCouplerArgs:
    diagram_id: str
    kv: float
    from_bus: str
    to_bus: str


@dataclass
class AddTransformerArgs:
    diagram_id: str
    kind: TransformerKind
    rated_mva: float
    kv_in: float
    kv_out: float
    tert_kv: float
    rating_mva: float
    vector_group: str
    percentZ: float
    ref_mva: Optional[float] = None
    
@dataclass
class AddBayArgs:
    diagram_id: str
    kind: BayKind
    kv: float
    bus: str

@dataclass
class AppendToBayArgs:
    diagram_id: str
    object_id: str 

@dataclass
class ConnectArgs:
    series: List[str] #list of IDs

@dataclass
class AddLineArgs:
    diagram_id: int
    kv: float
    kind : LineKind
    length_km: float
    thermal_A: float
    
@dataclass
class AddDisconnectorArgs:
    diagram_id: str
    kv: float
    kind: DisconnectorKind
    continuous_A: float

@dataclass
class ValidateArgs:
    pass

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
    ActionKind.APPEND_TO_BAY: AppendToBayArgs,
    ActionKind.CONNECT: ConnectArgs,
    ActionKind.ADD_LINE: AddLineArgs,
    ActionKind.ADD_DISCONNECTOR: AddDisconnectorArgs,
    ActionKind.VALIDATE: ValidateArgs,
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


