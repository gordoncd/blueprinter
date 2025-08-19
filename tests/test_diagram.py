"""
test_diagram.py

Test suite for the diagram.py module - tests the Diagram class and entity management.

Author: Generated Test Suite  
Date Created: 2025-08-19
"""
import pytest

from src.specgen.diagram import Diagram
from src.specgen.action import (
    Action, ActionKind, BusRole, Side, Scheme, BayKind, StepKind, 
    TransformerKind, ShuntKind,
    AddBusArgs, AddTransformerArgs, AddBayArgs,
    AppendStepArgs, AddBusConnectionArgs, AddShuntArgs, EmitSpecArgs
)


class TestDiagram:
    """Test the Diagram class functionality"""
    
    def test_diagram_initialization(self):
        """Test diagram initializes with empty collections"""
        diagram = Diagram()
        assert len(diagram.buses) == 0
        assert len(diagram.couplers) == 0
        assert len(diagram.transformers) == 0
        assert len(diagram.bays) == 0
        assert len(diagram.shunts) == 0
        assert len(diagram.bus_connections) == 0
        assert diagram.next_id == 0
    
    def test_get_next_id(self):
        """Test ID generation increments correctly"""
        diagram = Diagram()
        assert diagram.get_next_id() == 0
        assert diagram.get_next_id() == 1
        assert diagram.get_next_id() == 2
        assert diagram.next_id == 3


class TestDiagramActions:
    """Test applying actions to the diagram"""
    
    def test_add_bus_action(self):
        """Test adding a bus via action"""
        diagram = Diagram()
        args = AddBusArgs(diagram_id=1, kv=138.0, role=BusRole.MAIN)
        action = Action(ActionKind.ADD_BUS, args)
        
        bus_id = diagram.apply_action(action)
        
        assert bus_id == 0
        assert len(diagram.buses) == 1
        bus = diagram.buses[0]
        assert bus.kv == 138.0
        assert bus.role == BusRole.MAIN
        assert bus.diagram_id == 0
    
    def test_add_transformer_action(self):
        """Test adding a transformer via action"""
        diagram = Diagram()
        args = AddTransformerArgs(
            diagram_id=1,
            kind=TransformerKind.TWO_WINDING,
            hv_kv=138.0,
            lv_kv=13.8,
            rating_mva=50.0,
            vector_group="YNynd1"
        )
        action = Action(ActionKind.ADD_TRANSFORMER, args)
        
        tx_id = diagram.apply_action(action)
        
        assert tx_id == 0
        assert len(diagram.transformers) == 1
        tx = diagram.transformers[0]
        assert tx.kind == TransformerKind.TWO_WINDING
        assert tx.hv_kv == 138.0
        assert tx.lv_kv == 13.8
        assert tx.rating_mva == 50.0
        assert tx.vector_group == "YNynd1"
    
    def test_add_bay_action(self):
        """Test adding a bay via action"""
        diagram = Diagram()
        args = AddBayArgs(
            diagram_id=1,
            side=Side.HV,
            kv=138.0,
            kind=BayKind.TRANSFORMER_BAY,
            scheme=Scheme.SINGLE_BUSBAR,
            future=True
        )
        action = Action(ActionKind.ADD_BAY, args)
        
        bay_id = diagram.apply_action(action)
        
        assert bay_id == 0
        assert len(diagram.bays) == 1
        bay = diagram.bays[0]
        assert bay.side == Side.HV
        assert bay.kv == 138.0
        assert bay.kind == BayKind.TRANSFORMER_BAY
        assert bay.scheme == Scheme.SINGLE_BUSBAR
        assert bay.future is True
    
    def test_append_step_action(self):
        """Test appending steps to a bay"""
        diagram = Diagram()
        
        # First add a bay
        bay_args = AddBayArgs(
            diagram_id=1,
            side=Side.HV,
            kv=138.0,
            kind=BayKind.TRANSFORMER_BAY
        )
        bay_action = Action(ActionKind.ADD_BAY, bay_args)
        bay_id = diagram.apply_action(bay_action)
        
        # Then append steps
        step_args = AppendStepArgs(bay_id=bay_id, step_kind=StepKind.BREAKER)
        step_action = Action(ActionKind.APPEND_STEP, step_args)
        
        result = diagram.apply_action(step_action)
        
        assert result is True
        bay = diagram.bays[bay_id]
        assert len(bay.steps) == 1
        assert bay.steps[0] == StepKind.BREAKER
        
        # Add another step
        step_args2 = AppendStepArgs(bay_id=bay_id, step_kind=StepKind.CT)
        step_action2 = Action(ActionKind.APPEND_STEP, step_args2)
        diagram.apply_action(step_action2)
        
        assert len(bay.steps) == 2
        assert bay.steps[1] == StepKind.CT
    
    def test_append_step_invalid_bay(self):
        """Test appending step to non-existent bay raises error"""
        diagram = Diagram()
        step_args = AppendStepArgs(bay_id=999, step_kind=StepKind.BREAKER)
        step_action = Action(ActionKind.APPEND_STEP, step_args)
        
        with pytest.raises(ValueError, match="Bay with ID 999 not found"):
            diagram.apply_action(step_action)
    
    def test_add_shunt_action(self):
        """Test adding a shunt via action"""
        diagram = Diagram()
        args = AddShuntArgs(
            diagram_id=1,
            kind=ShuntKind.CAPACITOR,
            kv=13.8,
            mvar=25.0,
            bus_id=1,
            banks=2
        )
        action = Action(ActionKind.ADD_SHUNT, args)
        
        shunt_id = diagram.apply_action(action)
        
        assert shunt_id == 0
        assert len(diagram.shunts) == 1
        shunt = diagram.shunts[0]
        assert shunt.kind == ShuntKind.CAPACITOR
        assert shunt.kv == 13.8
        assert shunt.mvar == 25.0
        assert shunt.banks == 2
    
    def test_emit_spec_action(self):
        """Test emitting specification"""
        diagram = Diagram()
        
        # Add some entities first
        bus_args = AddBusArgs(diagram_id=1, kv=138.0, role=BusRole.MAIN)
        bus_action = Action(ActionKind.ADD_BUS, bus_args)
        diagram.apply_action(bus_action)
        
        # Emit spec
        emit_args = EmitSpecArgs()
        emit_action = Action(ActionKind.EMIT_SPEC, emit_args)
        
        spec = diagram.apply_action(emit_action)
        
        assert isinstance(spec, dict)
        assert "buses" in spec
        assert "couplers" in spec
        assert "transformers" in spec
        assert "bays" in spec
        assert "shunts" in spec
        assert "bus_connections" in spec
        assert len(spec["buses"]) == 1
        assert spec["buses"][0]["kv"] == 138.0
    
    def test_unknown_action_raises_error(self):
        """Test unknown action kind raises ValueError"""
        diagram = Diagram()
        
        # Create invalid action manually
        action = Action.__new__(Action)
        action.kind = "INVALID_ACTION"
        action.args = None
        
        with pytest.raises(ValueError, match="Unknown action kind"):
            diagram.apply_action(action)


class TestComplexWorkflow:
    """Test realistic electrical engineering workflows"""
    
    def test_simple_substation_workflow(self):
        """Test building a simple substation"""
        diagram = Diagram()
        
        # Add HV and LV buses
        hv_bus_args = AddBusArgs(diagram_id=1, kv=138.0, role=BusRole.MAIN)
        hv_bus_action = Action(ActionKind.ADD_BUS, hv_bus_args)
        hv_bus_id = diagram.apply_action(hv_bus_action)
        
        lv_bus_args = AddBusArgs(diagram_id=2, kv=13.8, role=BusRole.MAIN)
        lv_bus_action = Action(ActionKind.ADD_BUS, lv_bus_args)
        lv_bus_id = diagram.apply_action(lv_bus_action)
        
        # Add transformer
        tx_args = AddTransformerArgs(
            diagram_id=3,
            kind=TransformerKind.TWO_WINDING,
            hv_kv=138.0,
            lv_kv=13.8,
            rating_mva=50.0
        )
        tx_action = Action(ActionKind.ADD_TRANSFORMER, tx_args)
        tx_id = diagram.apply_action(tx_action)
        
        # Add transformer bay with protection
        bay_args = AddBayArgs(
            diagram_id=4,
            side=Side.HV,
            kv=138.0,
            kind=BayKind.TRANSFORMER_BAY
        )
        bay_action = Action(ActionKind.ADD_BAY, bay_args)
        bay_id = diagram.apply_action(bay_action)
        
        # Add protection steps
        steps = [StepKind.BUS_ISOLATOR, StepKind.BREAKER, StepKind.CT, StepKind.ARRESTER]
        for step in steps:
            step_args = AppendStepArgs(bay_id=bay_id, step_kind=step)
            step_action = Action(ActionKind.APPEND_STEP, step_args)
            diagram.apply_action(step_action)
        
        # Add bus connections
        hv_conn_args = AddBusConnectionArgs(
            diagram_id=5,
            bus_id=hv_bus_id,
            element_id=tx_id,
            element_side=Side.HV
        )
        hv_conn_action = Action(ActionKind.ADD_BUS_CONNECTION, hv_conn_args)
        diagram.apply_action(hv_conn_action)
        
        # Add shunt compensation
        shunt_args = AddShuntArgs(
            diagram_id=6,
            kind=ShuntKind.CAPACITOR,
            kv=13.8,
            mvar=25.0,
            bus_id=lv_bus_id
        )
        shunt_action = Action(ActionKind.ADD_SHUNT, shunt_args)
        diagram.apply_action(shunt_action)
        
        # Emit final specification
        emit_args = EmitSpecArgs()
        emit_action = Action(ActionKind.EMIT_SPEC, emit_args)
        spec = diagram.apply_action(emit_action)
        
        # Validate final state
        assert len(diagram.buses) == 2
        assert len(diagram.transformers) == 1
        assert len(diagram.bays) == 1
        assert len(diagram.bus_connections) == 1
        assert len(diagram.shunts) == 1
        
        # Validate bay has all protection steps
        bay = diagram.bays[bay_id]
        assert len(bay.steps) == 4
        assert StepKind.BREAKER in bay.steps
        assert StepKind.CT in bay.steps
        
        # Validate serialization
        assert len(spec["buses"]) == 2
        assert len(spec["transformers"]) == 1
        assert len(spec["bays"]) == 1
        assert spec["bays"][0]["steps"] == [
            StepKind.BUS_ISOLATOR, StepKind.BREAKER, StepKind.CT, StepKind.ARRESTER
        ]


if __name__ == "__main__":
    pytest.main([__file__])
