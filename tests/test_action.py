"""
test_action.py

Test suite for the action.py module in the specgen portion of the project.
Tests all action types, their validation, and edge cases.

Author: Generated Test Suite from GitHub Copilot
Date Created: 2025-08-19

"""
import pytest

from src.specgen.action import (
    # Enums
    BusRole, Side, Scheme, BayKind, StepKind, TransformerKind, 
    GroundingKind, ShuntKind, ActionKind,
    # Dataclasses
    AddBusArgs, AddCouplerArgs, AddTransformerArgs, AddBayArgs,
    AppendStepArgs, AddBusConnectionArgs, AddShuntArgs, EmitSpecArgs,
    # Main classes and functions
    Action, ACTION_ARG_SCHEMA, validate_action
)


class TestEnums:
    """Test all enum classes for correct values"""
    
    def test_bus_role_values(self):
        """Test BusRole enum values"""
        assert BusRole.MAIN == "main"
        assert BusRole.SECTION == "section"
        assert BusRole.TIE == "tie"
        assert len(BusRole) == 3
    
    def test_side_values(self):
        """Test Side enum values"""
        assert Side.HV == "HV"
        assert Side.LV == "LV"
        assert Side.TERT == "TERT"
        assert len(Side) == 3
    
    def test_scheme_values(self):
        """Test Scheme enum values"""
        assert Scheme.SINGLE_BUSBAR == "single_busbar"
        assert Scheme.DOUBLE_BUSBAR == "double_busbar"
        assert len(Scheme) == 2
    
    def test_bay_kind_values(self):
        """Test BayKind enum values"""
        assert BayKind.TRANSFORMER_BAY == "transformer_bay"
        assert BayKind.FEEDER_BAY == "feeder_bay"
        assert BayKind.LINE_BAY == "line_bay"
        assert BayKind.SHUNT_BAY == "shunt_bay"
        assert len(BayKind) == 4
    
    def test_step_kind_values(self):
        """Test StepKind enum values"""
        assert StepKind.BUS_ISOLATOR == "bus_isolator"
        assert StepKind.BREAKER == "breaker"
        assert StepKind.LINE_ISOLATOR == "line_isolator"
        assert StepKind.CT == "ct"
        assert StepKind.PT == "pt"
        assert StepKind.ARRESTER == "arrester"
        assert StepKind.PLACEHOLDER == "placeholder"
        assert len(StepKind) == 7
    
    def test_transformer_kind_values(self):
        """Test TransformerKind enum values"""
        assert TransformerKind.THREE_WINDING == "three_winding"
        assert TransformerKind.TWO_WINDING == "two_winding"
        assert TransformerKind.AUTOTRANSFORMER == "autotransformer"
        assert len(TransformerKind) == 3
    
    def test_grounding_kind_values(self):
        """Test GroundingKind enum values"""
        assert GroundingKind.SOLID == "solid"
        assert GroundingKind.NGR == "ngr"
        assert GroundingKind.REACTOR == "reactor"
        assert len(GroundingKind) == 3
    
    def test_shunt_kind_values(self):
        """Test ShuntKind enum values"""
        assert ShuntKind.CAPACITOR == "capacitor"
        assert len(ShuntKind) == 1
    
    def test_action_kind_values(self):
        """Test ActionKind enum values"""
        assert ActionKind.ADD_BUS == "ADD_BUS"
        assert ActionKind.ADD_COUPLER == "ADD_COUPLER"
        assert ActionKind.ADD_TRANSFORMER == "ADD_TRANSFORMER"
        assert ActionKind.ADD_BAY == "ADD_BAY"
        assert ActionKind.APPEND_STEP == "APPEND_STEP"
        assert ActionKind.ADD_BUS_CONNECTION == "ADD_BUS_CONNECTION"
        assert ActionKind.ADD_SHUNT == "ADD_SHUNT"
        assert ActionKind.EMIT_SPEC == "EMIT_SPEC"
        assert len(ActionKind) == 8


class TestDataclasses:
    """Test all action argument dataclasses"""
    
    def test_add_bus_args(self):
        """Test AddBusArgs dataclass"""
        args = AddBusArgs(
            diagram_id="test_diagram",
            kv=138.0,
            role=BusRole.MAIN
        )
        assert args.diagram_id == "test_diagram"
        assert args.kv == 138.0
        assert args.role == BusRole.MAIN
    
    def test_add_coupler_args(self):
        """Test AddCouplerArgs dataclass"""
        args = AddCouplerArgs(
            diagram_id="test_diagram",
            kv=138.0,
            busA_id="bus_1",
            busB_id="bus_2"
        )
        assert args.diagram_id == "test_diagram"
        assert args.kv == 138.0
        assert args.busA_id == "bus_1"
        assert args.busB_id == "bus_2"
    
    def test_add_transformer_args_minimal(self):
        """Test AddTransformerArgs with required fields only"""
        args = AddTransformerArgs(
            diagram_id="test_diagram",
            kind=TransformerKind.TWO_WINDING,
            hv_kv=138.0,
            lv_kv=13.8,
            rating_mva=50.0
        )
        assert args.diagram_id == "test_diagram"
        assert args.kind == TransformerKind.TWO_WINDING
        assert args.hv_kv == 138.0
        assert args.lv_kv == 13.8
        assert args.rating_mva == 50.0
        assert args.tert_kv is None
        assert args.vector_group is None
    
    def test_add_transformer_args_full(self):
        """Test AddTransformerArgs with all fields"""
        args = AddTransformerArgs(
            diagram_id="test_diagram",
            kind=TransformerKind.THREE_WINDING,
            hv_kv=138.0,
            lv_kv=13.8,
            rating_mva=50.0,
            tert_kv=4.16,
            vector_group="YNynd1",
            impedance_percent=8.5,
            ref_mva=100.0,
            grounding_hv=GroundingKind.SOLID,
            grounding_lv=GroundingKind.NGR,
            grounding_tert=GroundingKind.REACTOR
        )
        assert args.tert_kv == 4.16
        assert args.vector_group == "YNynd1"
        assert args.impedance_percent == 8.5
        assert args.ref_mva == 100.0
        assert args.grounding_hv == GroundingKind.SOLID
        assert args.grounding_lv == GroundingKind.NGR
        assert args.grounding_tert == GroundingKind.REACTOR
    
    def test_add_bay_args(self):
        """Test AddBayArgs dataclass"""
        args = AddBayArgs(
            diagram_id="test_diagram",
            side=Side.HV,
            kv=138.0,
            kind=BayKind.TRANSFORMER_BAY,
            scheme=Scheme.SINGLE_BUSBAR,
            future=True
        )
        assert args.diagram_id == "test_diagram"
        assert args.side == Side.HV
        assert args.kv == 138.0
        assert args.kind == BayKind.TRANSFORMER_BAY
        assert args.scheme == Scheme.SINGLE_BUSBAR
        assert args.future is True
    
    def test_append_step_args(self):
        """Test AppendStepArgs dataclass"""
        args = AppendStepArgs(
            bay_id="bay_1",
            step_kind=StepKind.BREAKER
        )
        assert args.bay_id == "bay_1"
        assert args.step_kind == StepKind.BREAKER
    
    def test_add_bus_connection_args(self):
        """Test AddBusConnectionArgs dataclass"""
        args = AddBusConnectionArgs(
            diagram_id="test_diagram",
            bus_id="bus_1",
            element_id="transformer_1",
            element_side=Side.HV
        )
        assert args.diagram_id == "test_diagram"
        assert args.bus_id == "bus_1"
        assert args.element_id == "transformer_1"
        assert args.element_side == Side.HV
    
    def test_add_shunt_args(self):
        """Test AddShuntArgs dataclass"""
        args = AddShuntArgs(
            diagram_id="test_diagram",
            kind=ShuntKind.CAPACITOR,
            kv=13.8,
            mvar=25.0,
            bus_id="bus_1",
            banks=3
        )
        assert args.diagram_id == "test_diagram"
        assert args.kind == ShuntKind.CAPACITOR
        assert args.kv == 13.8
        assert args.mvar == 25.0
        assert args.bus_id == "bus_1"
        assert args.banks == 3
    
    def test_emit_spec_args(self):
        """Test EmitSpecArgs dataclass"""
        args = EmitSpecArgs()
        assert isinstance(args, EmitSpecArgs)


class TestActionValidation:
    """Test the Action class and validation logic"""
    
    def test_valid_add_bus_action(self):
        """Test creating a valid ADD_BUS action"""
        args = AddBusArgs("test_diagram", 138.0, BusRole.MAIN)
        action = Action(ActionKind.ADD_BUS, args)
        assert action.kind == ActionKind.ADD_BUS
        assert action.args == args
    
    def test_valid_add_transformer_action(self):
        """Test creating a valid ADD_TRANSFORMER action"""
        args = AddTransformerArgs(
            diagram_id="test_diagram",
            kind=TransformerKind.TWO_WINDING,
            hv_kv=138.0,
            lv_kv=13.8,
            rating_mva=50.0
        )
        action = Action(ActionKind.ADD_TRANSFORMER, args)
        assert action.kind == ActionKind.ADD_TRANSFORMER
        assert action.args == args
    
    def test_valid_emit_spec_action(self):
        """Test creating a valid EMIT_SPEC action"""
        args = EmitSpecArgs()
        action = Action(ActionKind.EMIT_SPEC, args)
        assert action.kind == ActionKind.EMIT_SPEC
        assert action.args == args
    
    def test_invalid_action_kind(self):
        """Test validation fails with invalid action kind"""
        # Create a mock invalid action kind
        invalid_kind = "INVALID_ACTION"
        args = AddBusArgs("test_diagram", 138.0, BusRole.MAIN)
        
        # Manually create action to bypass enum validation
        action = Action.__new__(Action)
        action.kind = invalid_kind
        action.args = args
        
        with pytest.raises(ValueError, match="Invalid action kind"):
            validate_action(action)
    
    def test_invalid_args_type(self):
        """Test validation fails with wrong argument type"""
        # Use wrong args type for ADD_BUS
        wrong_args = AddCouplerArgs("test", 138.0, "busA", "busB")
        
        # Manually create action to bypass __post_init__
        action = Action.__new__(Action)
        action.kind = ActionKind.ADD_BUS
        action.args = wrong_args
        
        with pytest.raises(ValueError, match="Invalid args type"):
            validate_action(action)
    
    def test_action_arg_schema_completeness(self):
        """Test that ACTION_ARG_SCHEMA covers all ActionKind values"""
        for action_kind in ActionKind:
            assert action_kind in ACTION_ARG_SCHEMA
    
    def test_action_arg_schema_types(self):
        """Test that all schema entries map to correct dataclass types"""
        expected_mappings = {
            ActionKind.ADD_BUS: AddBusArgs,
            ActionKind.ADD_COUPLER: AddCouplerArgs,
            ActionKind.ADD_TRANSFORMER: AddTransformerArgs,
            ActionKind.ADD_BAY: AddBayArgs,
            ActionKind.APPEND_STEP: AppendStepArgs,
            ActionKind.ADD_BUS_CONNECTION: AddBusConnectionArgs,
            ActionKind.ADD_SHUNT: AddShuntArgs,
            ActionKind.EMIT_SPEC: EmitSpecArgs,
        }
        
        for action_kind, expected_type in expected_mappings.items():
            assert ACTION_ARG_SCHEMA[action_kind] == expected_type


class TestActionWorkflows:
    """Test realistic action workflows"""
    
    def test_simple_transformer_workflow(self):
        """Test a simple transformer substation workflow"""
        actions = []
        
        # Add HV bus
        hv_bus_args = AddBusArgs("diagram1", 138.0, BusRole.MAIN)
        actions.append(Action(ActionKind.ADD_BUS, hv_bus_args))
        
        # Add LV bus
        lv_bus_args = AddBusArgs("diagram1", 13.8, BusRole.MAIN)
        actions.append(Action(ActionKind.ADD_BUS, lv_bus_args))
        
        # Add transformer
        tx_args = AddTransformerArgs(
            diagram_id="diagram1",
            kind=TransformerKind.TWO_WINDING,
            hv_kv=138.0,
            lv_kv=13.8,
            rating_mva=50.0
        )
        actions.append(Action(ActionKind.ADD_TRANSFORMER, tx_args))
        
        # Add transformer bay
        bay_args = AddBayArgs(
            diagram_id="diagram1",
            side=Side.HV,
            kv=138.0,
            kind=BayKind.TRANSFORMER_BAY
        )
        actions.append(Action(ActionKind.ADD_BAY, bay_args))
        
        # Add protection steps
        step_args = AppendStepArgs("bay1", StepKind.BREAKER)
        actions.append(Action(ActionKind.APPEND_STEP, step_args))
        
        step_args = AppendStepArgs("bay1", StepKind.CT)
        actions.append(Action(ActionKind.APPEND_STEP, step_args))
        
        # Emit specification
        emit_args = EmitSpecArgs()
        actions.append(Action(ActionKind.EMIT_SPEC, emit_args))
        
        # All actions should be valid
        for action in actions:
            assert isinstance(action, Action)
            assert action.kind in ActionKind
    
    def test_three_winding_transformer_workflow(self):
        """Test three-winding transformer with tertiary"""
        # Add three-winding transformer
        tx_args = AddTransformerArgs(
            diagram_id="diagram1",
            kind=TransformerKind.THREE_WINDING,
            hv_kv=138.0,
            lv_kv=13.8,
            rating_mva=75.0,
            tert_kv=4.16,
            vector_group="YNynd1",
            grounding_hv=GroundingKind.SOLID,
            grounding_lv=GroundingKind.NGR,
            grounding_tert=GroundingKind.REACTOR
        )
        action = Action(ActionKind.ADD_TRANSFORMER, tx_args)
        
        assert action.args.tert_kv == 4.16
        assert action.args.vector_group == "YNynd1"
        assert action.args.grounding_hv == GroundingKind.SOLID
    
    def test_shunt_compensation_workflow(self):
        """Test adding shunt compensation"""
        # Add capacitor bank
        shunt_args = AddShuntArgs(
            diagram_id="diagram1",
            kind=ShuntKind.CAPACITOR,
            kv=13.8,
            mvar=25.0,
            bus_id="lv_bus",
            banks=2
        )
        action = Action(ActionKind.ADD_SHUNT, shunt_args)
        
        assert action.args.kind == ShuntKind.CAPACITOR
        assert action.args.mvar == 25.0
        assert action.args.banks == 2


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_voltage_values(self):
        """Test handling of zero voltage values"""
        args = AddBusArgs("test", 0.0, BusRole.MAIN)
        action = Action(ActionKind.ADD_BUS, args)
        assert action.args.kv == 0.0
    
    def test_negative_voltage_values(self):
        """Test handling of negative voltage values"""
        args = AddBusArgs("test", -138.0, BusRole.MAIN)
        action = Action(ActionKind.ADD_BUS, args)
        assert action.args.kv == -138.0
    
    def test_very_large_mva_rating(self):
        """Test handling of very large MVA ratings"""
        args = AddTransformerArgs(
            diagram_id="test",
            kind=TransformerKind.TWO_WINDING,
            hv_kv=500.0,
            lv_kv=138.0,
            rating_mva=1000.0
        )
        action = Action(ActionKind.ADD_TRANSFORMER, args)
        assert action.args.rating_mva == 1000.0
    
    def test_empty_string_ids(self):
        """Test handling of empty string IDs"""
        args = AddBusArgs("", 138.0, BusRole.MAIN)
        action = Action(ActionKind.ADD_BUS, args)
        assert action.args.diagram_id == ""
    
    def test_none_optional_values(self):
        """Test that optional values can be None"""
        args = AddTransformerArgs(
            diagram_id="test",
            kind=TransformerKind.TWO_WINDING,
            hv_kv=138.0,
            lv_kv=13.8,
            rating_mva=50.0,
            tert_kv=None,
            vector_group=None,
            impedance_percent=None
        )
        action = Action(ActionKind.ADD_TRANSFORMER, args)
        assert action.args.tert_kv is None
        assert action.args.vector_group is None
        assert action.args.impedance_percent is None


if __name__ == "__main__":
    pytest.main([__file__])
