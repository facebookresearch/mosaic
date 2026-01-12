# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from later.unittest import TestCase
from mosaic.cmd.entry_point import get_memory_profile
from mosaic.cmd.get_memory_profile import main
from mosaic.libmosaic.utils.data_utils import (
    AllocationType,
    Frame,
    MemoryUsage,
    TraceEvent,
)


def snapshot_path_str() -> str:
    """Helper function to get test snapshot path"""
    return "test_snapshot.pickle"


class TestCustomProfiling(TestCase):
    """Unit tests for custom profiling functionality"""

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_custom_pattern_matching_single_pattern(self) -> None:
        """Test custom pattern matching with a single pattern"""
        custom_rules = {"my_kernel": "my_kernel_.*"}

        frames = [Frame(name="my_kernel_forward", filename="model.py", line=10)]
        alloc_type, category = AllocationType.from_frame_stack_with_custom(
            frames, custom_rules
        )

        self.assertEqual(alloc_type, AllocationType.CUSTOM)
        self.assertEqual(category, "my_kernel")

    def test_custom_pattern_fallback_to_existing_logic(self) -> None:
        """Test that custom patterns fall back to existing logic when no match"""
        custom_rules = {"nonmatch": "nonexistent_pattern"}
        frames = [Frame(name="forward", filename="model.py", line=10)]

        alloc_type, category = AllocationType.from_frame_stack_with_custom(
            frames, custom_rules
        )

        self.assertEqual(alloc_type, AllocationType.ACTIVATION)
        self.assertIsNone(category)

    def test_custom_pattern_no_rules_provided(self) -> None:
        """Test behavior when no custom rules are provided"""
        frames = [Frame(name="forward", filename="model.py", line=10)]

        alloc_type, category = AllocationType.from_frame_stack_with_custom(frames, None)

        self.assertEqual(alloc_type, AllocationType.ACTIVATION)
        self.assertIsNone(category)

    def test_custom_pattern_invalid_regex(self) -> None:
        """Test that invalid regex patterns are handled gracefully"""
        custom_rules = {"bad_regex": "[invalid"}

        frames = [Frame(name="test_function", filename="test.py", line=10)]
        alloc_type, category = AllocationType.from_frame_stack_with_custom(
            frames, custom_rules
        )

        # Should fall back to existing logic since regex is invalid
        self.assertEqual(alloc_type, AllocationType.UNKNOWN)
        self.assertIsNone(category)

    def test_custom_pattern_first_match_wins(self) -> None:
        """Test that first matching pattern wins when multiple patterns match"""
        custom_rules = {
            "first_pattern": "test_.*",
            "second_pattern": ".*function",
        }

        frames = [Frame(name="test_function", filename="test.py", line=10)]
        alloc_type, category = AllocationType.from_frame_stack_with_custom(
            frames, custom_rules
        )

        self.assertEqual(alloc_type, AllocationType.CUSTOM)
        self.assertEqual(category, "first_pattern")

    def test_hierarchical_categorization(self) -> None:
        """Test hierarchical categorization with specific → general ordering"""
        custom_rules = {
            "fsdp_forward": "fsdp.*forward",  # Most specific
            "fsdp_general": "fsdp",  # Less specific
            "pytorch_forward": ".*forward",  # General forward operations
            "all_operations": ".*",  # Catch-all
        }

        # Test most specific pattern wins
        frames = [Frame(name="fsdp_linear_forward", filename="model.py", line=10)]
        alloc_type, category = AllocationType.from_frame_stack_with_custom(
            frames, custom_rules
        )
        self.assertEqual(alloc_type, AllocationType.CUSTOM)
        self.assertEqual(category, "fsdp_forward")

        # Test second-level specificity
        frames = [Frame(name="fsdp_backward", filename="model.py", line=20)]
        alloc_type, category = AllocationType.from_frame_stack_with_custom(
            frames, custom_rules
        )
        self.assertEqual(alloc_type, AllocationType.CUSTOM)
        self.assertEqual(category, "fsdp_general")

        # Test third-level specificity
        frames = [Frame(name="linear_forward", filename="model.py", line=30)]
        alloc_type, category = AllocationType.from_frame_stack_with_custom(
            frames, custom_rules
        )
        self.assertEqual(alloc_type, AllocationType.CUSTOM)
        self.assertEqual(category, "pytorch_forward")

        # Test catch-all
        frames = [Frame(name="random_function", filename="other.py", line=40)]
        alloc_type, category = AllocationType.from_frame_stack_with_custom(
            frames, custom_rules
        )
        self.assertEqual(alloc_type, AllocationType.CUSTOM)
        self.assertEqual(category, "all_operations")

    def test_trace_event_from_raw_with_custom_rules(self) -> None:
        """Test TraceEvent.from_raw with custom rules"""
        custom_rules = {"test_category": "test_.*"}

        raw_event = {
            "action": "alloc",
            "addr": 123,
            "size": 1000,
            "stream": 0,
            "time_us": 1000,
            "frames": [{"name": "test_function", "filename": "test.py", "line": 10}],
        }

        evt = TraceEvent.from_raw(raw_event, "test_annotation", custom_rules)

        self.assertEqual(evt.action, "alloc")
        self.assertEqual(evt.addr, 123)
        self.assertEqual(evt.size, 1000)
        self.assertEqual(evt.classification, AllocationType.CUSTOM)
        self.assertEqual(evt.custom_category, "test_category")
        self.assertEqual(evt.annotation, "test_annotation")

    def test_trace_event_from_raw_fallback(self) -> None:
        """Test TraceEvent.from_raw falls back when no custom match"""
        custom_rules = {"no_match": "no_match_pattern"}

        raw_event = {
            "action": "alloc",
            "addr": 123,
            "size": 1000,
            "stream": 0,
            "time_us": 1000,
            "frames": [{"name": "forward", "filename": "model.py", "line": 10}],
        }

        evt = TraceEvent.from_raw(raw_event, "test_annotation", custom_rules)

        self.assertEqual(evt.classification, AllocationType.ACTIVATION)
        self.assertEqual(evt.custom_category, "unknown")

    def test_memory_usage_custom_tracking(self) -> None:
        """Test MemoryUsage tracks custom categories correctly"""
        memory_usage = MemoryUsage(save_profile=True)

        # Create mock TraceEvent with custom category
        evt = TraceEvent(
            action="alloc",
            addr=123,
            size=1000,
            stream=0,
            time_us=1000,
            classification=AllocationType.CUSTOM,
            custom_category="test_kernel",
        )

        memory_usage.update(evt, ["custom"])
        self.assertEqual(memory_usage.per_custom_alloc_sum["test_kernel"], 1000)

        # Test freeing memory
        free_evt = TraceEvent(
            action="free_completed",
            addr=123,
            size=1000,
            stream=0,
            time_us=2000,
            classification=AllocationType.CUSTOM,
            custom_category="test_kernel",
        )

        memory_usage.update(free_evt, ["custom"])
        self.assertEqual(memory_usage.per_custom_alloc_sum["test_kernel"], 0)


class TestCustomProfilingCLI(TestCase):
    """Integration tests for custom profiling CLI functionality"""

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_cli_validation_missing_custom_profile(self) -> None:
        """Test CLI validation when custom profile is missing"""
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--snapshot",
                "test.pickle",
                "--profile",
                "custom",
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("--custom-profile required when --profile=custom", result.output)

    def test_cli_validation_custom_profile_with_wrong_mode(self) -> None:
        """Test CLI validation when custom profile is used with non-custom mode"""
        from click.testing import CliRunner

        runner = CliRunner()
        result = runner.invoke(
            main,
            [
                "--snapshot",
                "test.pickle",
                "--profile",
                "categories",
                "--custom-profile",
                '{"test": "pattern"}',
            ],
        )

        self.assertNotEqual(result.exit_code, 0)
        self.assertIn(
            "--custom-profile only valid with --profile=custom", result.output
        )

    def test_invalid_json_custom_profile(self) -> None:
        """Test error handling for invalid JSON in custom profile"""
        with self.assertRaises(ValueError) as context:
            get_memory_profile(
                snapshot="test.pickle",
                out_path="test.html",
                profile="custom",
                custom_profile="invalid json",
            )

        self.assertIn("Failed to parse custom profile", str(context.exception))

    def test_non_dict_custom_profile(self) -> None:
        """Test error handling when custom profile is not a dictionary"""
        with self.assertRaises(ValueError) as context:
            get_memory_profile(
                snapshot="test.pickle",
                out_path="test.html",
                profile="custom",
                custom_profile='"not a dict"',
            )

        self.assertIn(
            "Custom profile must be a JSON dictionary", str(context.exception)
        )

    def test_invalid_dict_values_custom_profile(self) -> None:
        """Test error handling when custom profile has non-string values"""
        with self.assertRaises(ValueError) as context:
            get_memory_profile(
                snapshot="test.pickle",
                out_path="test.html",
                profile="custom",
                custom_profile='{"key": 123}',
            )

        self.assertIn(
            "Custom profile dictionary keys and values must be strings",
            str(context.exception),
        )


class TestCustomProfilingEdgeCases(TestCase):
    """Tests for edge cases in custom profiling"""

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_empty_custom_profile_dictionary(self) -> None:
        """Test behavior with empty custom profile dictionary"""
        frames = [Frame(name="forward", filename="model.py", line=10)]

        alloc_type, category = AllocationType.from_frame_stack_with_custom(frames, {})

        # Should fall back to existing logic
        self.assertEqual(alloc_type, AllocationType.ACTIVATION)
        self.assertIsNone(category)

    def test_unicode_category_names(self) -> None:
        """Test custom profiling with Unicode category names"""
        custom_rules = {"测试类别": "test_.*", "función": "function_.*"}

        frames = [Frame(name="test_function", filename="test.py", line=10)]
        alloc_type, category = AllocationType.from_frame_stack_with_custom(
            frames, custom_rules
        )

        self.assertEqual(alloc_type, AllocationType.CUSTOM)
        self.assertEqual(category, "测试类别")

    def test_special_regex_characters_in_patterns(self) -> None:
        """Test custom profiling with special regex characters"""
        custom_rules = {"brackets": r"\[.*\]", "parens": r"\(.*\)"}

        # Test brackets
        frames = [Frame(name="[bracket_function]", filename="test.py", line=10)]
        alloc_type, category = AllocationType.from_frame_stack_with_custom(
            frames, custom_rules
        )

        self.assertEqual(alloc_type, AllocationType.CUSTOM)
        self.assertEqual(category, "brackets")

        # Test parentheses
        frames = [Frame(name="(paren_function)", filename="test.py", line=10)]
        alloc_type, category = AllocationType.from_frame_stack_with_custom(
            frames, custom_rules
        )

        self.assertEqual(alloc_type, AllocationType.CUSTOM)
        self.assertEqual(category, "parens")


class TestOmegaConfIntegration(TestCase):
    """Tests for OmegaConf integration with custom profiling"""

    def setUp(self) -> None:
        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()

    def test_structured_yaml_config_parsing(self) -> None:
        """Test parsing structured YAML config with OmegaConf"""
        import tempfile

        from mosaic.cmd.entry_point import get_memory_profile

        yaml_config = """
        rules:
          - name: "fsdp_forward"
            pattern: "fsdp.*forward"
            description: "FSDP forward operations"
            priority: 1
          - name: "fsdp_backward"
            pattern: "fsdp.*backward"
            description: "FSDP backward operations"
            priority: 2
          - name: "general"
            pattern: ".*"
            description: "All other operations"
            priority: 999
        """

        # Test that the config is parseable (this will validate structure)
        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="w+", delete=True
        ) as output_file:
            # This should not raise an exception for valid YAML
            try:
                get_memory_profile(
                    snapshot=snapshot_path_str(),
                    out_path=output_file.name,
                    profile="custom",
                    custom_profile=yaml_config,
                )
            except Exception as e:
                # If it fails, it should be due to missing snapshot file, not config parsing
                self.assertIn("snapshot", str(e).lower())

    def test_dataclass_validation(self) -> None:
        """Test that dataclass validation works for custom profiles"""
        from mosaic.cmd.entry_point import CustomProfile, CustomProfileRule

        # Test valid dataclass creation
        profile = CustomProfile(
            rules=[
                CustomProfileRule(
                    name="test_category",
                    pattern="test_.*",
                    description="Test operations",
                )
            ]
        )

        rules_dict = profile.to_dict()
        self.assertEqual(rules_dict["test_category"], "test_.*")

    def test_backward_compatibility_json(self) -> None:
        """Test that simple JSON format still works"""
        import tempfile

        from mosaic.cmd.entry_point import get_memory_profile

        json_config = '{"test_category": "test_.*", "other": "other_.*"}'

        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="w+", delete=True
        ) as output_file:
            try:
                get_memory_profile(
                    snapshot=snapshot_path_str(),
                    out_path=output_file.name,
                    profile="custom",
                    custom_profile=json_config,
                )
            except Exception as e:
                # Should fail on missing snapshot, not config parsing
                self.assertIn("snapshot", str(e).lower())

    def test_invalid_yaml_config_error_handling(self) -> None:
        """Test proper error handling for invalid YAML configs"""
        import tempfile

        from mosaic.cmd.entry_point import get_memory_profile

        invalid_yaml = """
        rules:
          - name: "test"
            invalid_field: "should_not_exist"
        """

        with tempfile.NamedTemporaryFile(
            suffix=".html", mode="w+", delete=True
        ) as output_file:
            with self.assertRaises(ValueError) as context:
                get_memory_profile(
                    snapshot=snapshot_path_str(),
                    out_path=output_file.name,
                    profile="custom",
                    custom_profile=invalid_yaml,
                )

            self.assertIn("Failed to parse custom profile", str(context.exception))
