"""
diagram.py

This module defines the entities that are being dispatched by the action space
it keeps track of the current state of the diagram and the actions that can be performed on it
it assigns ids to each entity, applies validation/feedback, and provides a way to serialize the diagram to a JSON format

structure mirrors action.py, but focuses on the diagram entities rather than actions

Author: Gordon Doore
Date Created: 2025-08-19
Last Modified: 2025-08-19

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
from .action import (
    Action, ActionKind, BusRole, Side, Scheme, BayKind, StepKind, 
    TransformerKind, GroundingKind, ShuntKind
)


@dataclass
class Bus:
    diagram_id: int
    role: BusRole
    kv: float

@dataclass
class Coupler:
    diagram_id: int
    kv: float
    busA_id: int
    busB_id: int

@dataclass
class Transformer:
    diagram_id: int
    kind: TransformerKind
    hv_kv: float
    lv_kv: float
    rating_mva: float
    tert_kv: Optional[float] = None
    vector_group: Optional[str] = None
    impedance_percent: Optional[float] = None
    ref_mva: Optional[float] = None
    grounding_hv: Optional[GroundingKind] = None
    grounding_lv: Optional[GroundingKind] = None
    grounding_tert: Optional[GroundingKind] = None

@dataclass
class Bay:
    diagram_id: int
    side: Side
    kv: float
    kind: BayKind
    scheme: Optional[Scheme] = None
    future: bool = False
    steps: Optional[list[StepKind]] = None

    def append_step(self, step_kind: StepKind):
        if self.steps is None:
            self.steps = []
        self.steps.append(step_kind)
    
@dataclass
class BusConnection:
    diagram_id: int
    bus_id: int
    element_id: int  # ID of transformer, bay, or shunt being connected
    element_side: Optional[Side] = None  # Side of the element to connect to the bus

@dataclass
class Shunt:
    diagram_id: int
    kind: ShuntKind
    kv: float
    mvar: float
    bus_id: int
    banks: Optional[int] = None  # Number of banks for the shunt, if applicable



class Diagram:
    """
    Represents the current state of the diagram.
    Contains all entities and their relationships.
    Provides methods to manipulate and serialize the diagram.
    """
    def __init__(self):
        self.buses: Dict[int, Bus] = {}
        self.couplers: Dict[int, Coupler] = {}
        self.transformers: Dict[int, Transformer] = {}
        self.bays: Dict[int, Bay] = {}
        self.shunts: Dict[int, Shunt] = {}
        self.bus_connections: Dict[int, BusConnection] = {}
        self.next_id: int = 0

    def apply_action(self, action: Action):
        """Apply an action to modify the diagram state."""
        if action.kind == ActionKind.ADD_BUS:
            bus = Bus(
                diagram_id=-1, 
                role=action.args.role, 
                kv=action.args.kv
            )
            self.add_bus(bus)
            return bus.diagram_id
            
        elif action.kind == ActionKind.ADD_COUPLER:
            coupler = Coupler(
                diagram_id=-1, 
                kv=action.args.kv, 
                busA_id=action.args.busA_id, 
                busB_id=action.args.busB_id
            )
            self.add_coupler(coupler)
            return coupler.diagram_id
            
        elif action.kind == ActionKind.ADD_TRANSFORMER:
            transformer = Transformer(
                diagram_id=-1,
                kind=action.args.kind,
                hv_kv=action.args.hv_kv,
                lv_kv=action.args.lv_kv,
                rating_mva=action.args.rating_mva,
                tert_kv=action.args.tert_kv,
                vector_group=action.args.vector_group,
                impedance_percent=action.args.impedance_percent,
                ref_mva=action.args.ref_mva,
                grounding_hv=action.args.grounding_hv,
                grounding_lv=action.args.grounding_lv,
                grounding_tert=action.args.grounding_tert
            )
            self.add_transformer(transformer)
            return transformer.diagram_id
            
        elif action.kind == ActionKind.ADD_BAY:
            bay = Bay(
                diagram_id=-1,
                side=action.args.side,
                kv=action.args.kv,
                kind=action.args.kind,
                scheme=action.args.scheme,
                future=action.args.future
            )
            self.add_bay(bay)
            return bay.diagram_id
            
        elif action.kind == ActionKind.APPEND_STEP:
            bay = self.bays.get(action.args.bay_id)
            if bay:
                bay.append_step(action.args.step_kind)
                return True
            else:
                raise ValueError(f"Bay with ID {action.args.bay_id} not found")
                
        elif action.kind == ActionKind.ADD_BUS_CONNECTION:
            bus_connection = BusConnection(
                diagram_id=-1,
                bus_id=action.args.bus_id,
                element_id=action.args.element_id,
                element_side=action.args.element_side
            )
            self.add_bus_connection(bus_connection)
            return bus_connection.diagram_id
            
        elif action.kind == ActionKind.ADD_SHUNT:
            shunt = Shunt(
                diagram_id=-1,
                kind=action.args.kind,
                kv=action.args.kv,
                mvar=action.args.mvar,
                bus_id=action.args.bus_id,
                banks=action.args.banks
            )
            self.add_shunt(shunt)
            return shunt.diagram_id
            
        elif action.kind == ActionKind.EMIT_SPEC:
            return self.serialize()
            
        else:
            raise ValueError(f"Unknown action kind: {action.kind}")

    def get_next_id(self) -> int:
        current_id = self.next_id
        self.next_id += 1
        return current_id

    def add_bus(self, bus: Bus):
        bus.diagram_id = self.get_next_id()
        self.buses[bus.diagram_id] = bus

    def add_coupler(self, coupler: Coupler):
        coupler.diagram_id = self.get_next_id()
        self.couplers[coupler.diagram_id] = coupler

    def add_transformer(self, transformer: Transformer):
        transformer.diagram_id = self.get_next_id()
        self.transformers[transformer.diagram_id] = transformer

    def add_bay(self, bay: Bay):
        bay.diagram_id = self.get_next_id()
        self.bays[bay.diagram_id] = bay

    def add_shunt(self, shunt: Shunt):
        shunt.diagram_id = self.get_next_id()
        self.shunts[shunt.diagram_id] = shunt

    def add_bus_connection(self, bus_connection: BusConnection):
        bus_connection.diagram_id = self.get_next_id()
        self.bus_connections[bus_connection.diagram_id] = bus_connection


    def serialize(self) -> Dict[str, Any]:
        """
        Serialize the diagram to a JSON-compatible dictionary.
        """
        return {
            "buses": [vars(bus) for bus in self.buses.values()],
            "couplers": [vars(coupler) for coupler in self.couplers.values()],
            "transformers": [vars(transformer) for transformer in self.transformers.values()],
            "bays": [vars(bay) for bay in self.bays.values()],
            "shunts": [vars(shunt) for shunt in self.shunts.values()],
            "bus_connections": [vars(conn) for conn in self.bus_connections.values()],
        }

    

