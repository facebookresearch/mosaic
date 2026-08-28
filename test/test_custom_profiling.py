# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from later.unittest import TestCase
from mosaic.cmd.entry_point import get_memory_profile
from mosaic.cmd.get_memory_profile import main
from mosaic.libmosaic.utils.data_utils import (
    _any_filename_contains,
    _any_filename_matches_regex,
    _any_frame_name_contains,
    _any_frame_name_in,
    _any_frame_name_startswith,
    _no_frame_name_in,
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


class TestFrameStackPredicateHelpers(TestCase):
    """Unit tests for the small frame-stack predicate helpers used by the
    AllocationType rule table."""

    def setUp(self) -> None:
        super().setUp()
        # A small synthetic stack used by every assertion below.
        self.stack: list[Frame] = [
            Frame(name="forward", filename="model.py", line=10),
            Frame(name="custom_adamw", filename="optim.py", line=20),
            Frame(name="my_kernel_v2", filename="kernel.cu", line=30),
        ]

    def test_any_filename_contains(self) -> None:
        self.assertTrue(_any_filename_contains(self.stack, "model.py"))
        self.assertTrue(_any_filename_contains(self.stack, "optim"))
        self.assertFalse(_any_filename_contains(self.stack, "missing.py"))
        # Empty stack always returns False.
        self.assertFalse(_any_filename_contains([], "anything"))

    def test_any_filename_matches_regex(self) -> None:
        import re

        cu_pattern = re.compile(r"\.cu$")
        py_pattern = re.compile(r"^model\.py$")
        miss_pattern = re.compile(r"^never$")
        self.assertTrue(_any_filename_matches_regex(self.stack, cu_pattern))
        self.assertTrue(_any_filename_matches_regex(self.stack, py_pattern))
        self.assertFalse(_any_filename_matches_regex(self.stack, miss_pattern))
        self.assertFalse(_any_filename_matches_regex([], cu_pattern))

    def test_any_frame_name_in(self) -> None:
        self.assertTrue(
            _any_frame_name_in(self.stack, frozenset({"custom_adamw", "other"}))
        )
        self.assertFalse(
            _any_frame_name_in(self.stack, frozenset({"never", "missing"}))
        )
        self.assertFalse(_any_frame_name_in([], frozenset({"forward"})))

    def test_any_frame_name_contains(self) -> None:
        self.assertTrue(_any_frame_name_contains(self.stack, "forward"))
        self.assertTrue(_any_frame_name_contains(self.stack, "kernel"))
        self.assertFalse(_any_frame_name_contains(self.stack, "missing"))
        self.assertFalse(_any_frame_name_contains([], "forward"))

    def test_any_frame_name_startswith(self) -> None:
        self.assertTrue(_any_frame_name_startswith(self.stack, "forward"))
        self.assertTrue(_any_frame_name_startswith(self.stack, "my_kernel"))
        # "kernel" appears mid-name, so startswith should not match.
        self.assertFalse(_any_frame_name_startswith(self.stack, "kernel"))
        self.assertFalse(_any_frame_name_startswith([], "forward"))

    def test_no_frame_name_in(self) -> None:
        # Inverse of _any_frame_name_in.
        self.assertFalse(_no_frame_name_in(self.stack, frozenset({"custom_adamw"})))
        self.assertTrue(_no_frame_name_in(self.stack, frozenset({"never"})))
        # Empty stack means no frame is in the set, so True.
        self.assertTrue(_no_frame_name_in([], frozenset({"forward"})))


class TestAllocationTypeEnumExtensions(TestCase):
    """Tests for the EMBEDDING and COMPILE enum members."""

    def test_embedding_member_exists_with_expected_value(self) -> None:
        self.assertEqual(AllocationType.EMBEDDING.value, 10)
        self.assertEqual(AllocationType.EMBEDDING.name, "EMBEDDING")

    def test_compile_member_exists_with_expected_value(self) -> None:
        self.assertEqual(AllocationType.COMPILE.value, 11)
        self.assertEqual(AllocationType.COMPILE.name, "COMPILE")

    def test_per_category_alloc_sum_accepts_embedding_and_compile(self) -> None:
        """The MemoryUsage per-category dict is keyed by AllocationType, so
        new enum values must be usable as keys without any extra wiring."""
        memory_usage = MemoryUsage(save_profile=True)

        for cat in (AllocationType.EMBEDDING, AllocationType.COMPILE):
            evt = TraceEvent(
                action="alloc",
                addr=hash(cat),
                size=4096,
                stream=0,
                time_us=0,
                classification=cat,
            )
            memory_usage.update(evt, ["categories"])

        self.assertEqual(
            memory_usage.per_category_alloc_sum[AllocationType.EMBEDDING], 4096
        )
        self.assertEqual(
            memory_usage.per_category_alloc_sum[AllocationType.COMPILE], 4096
        )


class TestBackwardCategorization(TestCase):
    """BACKWARD-domain rules: C++ autograd, DDP reducer, grad-scale helpers.

    Frame names use the demangled C++ symbol form with parameter signatures,
    so prefix-vs-exact-match bugs cannot hide.
    """

    def _stack(self, name: str, filename: str = "<unknown>") -> list[Frame]:
        return [Frame(name=name, filename=filename, line=0)]

    def test_autograd_engine_evaluate_function_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack(
                    "torch::autograd::Engine::evaluate_function("
                    "std::shared_ptr<torch::autograd::GraphTask>&, "
                    "torch::autograd::Node*, torch::autograd::InputBuffer&, "
                    "std::shared_ptr<torch::autograd::ReadyQueue> const&)"
                )
            ),
            AllocationType.BACKWARD,
        )

    def test_autograd_python_engine_thread_init_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack(
                    "torch::autograd::python::PythonEngine::thread_init(int, "
                    "std::shared_ptr<torch::autograd::ReadyQueue> const&, bool)"
                )
            ),
            AllocationType.BACKWARD,
        )

    def test_autograd_node_operator_call_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack(
                    "torch::autograd::Node::operator()("
                    "std::vector<at::Tensor, std::allocator<at::Tensor>>&&)"
                )
            ),
            AllocationType.BACKWARD,
        )

    def test_autograd_pynode_apply_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack(
                    "torch::autograd::PyNode::apply("
                    "std::vector<at::Tensor, std::allocator<at::Tensor>>&&)"
                )
            ),
            AllocationType.BACKWARD,
        )

    def test_autograd_cppnode_apply_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack(
                    "torch::autograd::CppNode<SomeBackwardFn>::apply("
                    "std::vector<at::Tensor, std::allocator<at::Tensor>>&&)"
                )
            ),
            AllocationType.BACKWARD,
        )

    def test_autograd_graph_task_post_processing_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack("torch::autograd::GraphTask::exec_post_processing()")
            ),
            AllocationType.BACKWARD,
        )

    def test_autograd_accumulate_grad_apply_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack(
                    "torch::autograd::AccumulateGrad::apply("
                    "std::vector<at::Tensor, std::allocator<at::Tensor>>&&)"
                )
            ),
            AllocationType.BACKWARD,
        )

    def test_autograd_delete_node_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack("torch::autograd::deleteNode(torch::autograd::Node*)")
            ),
            AllocationType.BACKWARD,
        )

    def test_autograd_lambda_post_hook_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack(
                    "torch::autograd::utils::LambdaPostHook::operator()("
                    "std::vector<at::Tensor, std::allocator<at::Tensor>> const&, "
                    "std::vector<at::Tensor, std::allocator<at::Tensor>> const&)"
                )
            ),
            AllocationType.BACKWARD,
        )

    def test_ddp_reducer_mark_variable_ready_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack("c10d::Reducer::mark_variable_ready(unsigned long)")
            ),
            AllocationType.BACKWARD,
        )

    def test_ddp_reducer_mark_variable_ready_dense_is_backward(self) -> None:
        # Overload variant exact-name matching would miss.
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack("c10d::Reducer::mark_variable_ready_dense(unsigned long)")
            ),
            AllocationType.BACKWARD,
        )

    def test_ddp_reducer_autograd_hook_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack("c10d::Reducer::autograd_hook(unsigned long)")
            ),
            AllocationType.BACKWARD,
        )

    def test_ddp_reducer_rebuild_buckets_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(
                self._stack("c10d::Reducer::rebuild_buckets()")
            ),
            AllocationType.BACKWARD,
        )

    def test_python_grad_scale_helper_is_backward(self) -> None:
        self.assertEqual(
            AllocationType.from_frame_stack(self._stack("_instantiate_filtered_grads")),
            AllocationType.BACKWARD,
        )

    def test_existing_clip_grad_norm_still_works(self) -> None:
        # Regression: pre-existing rule must still match after new rules are
        # inserted ahead of it.
        self.assertEqual(
            AllocationType.from_frame_stack(self._stack("clip_grad_norm_")),
            AllocationType.BACKWARD,
        )


class TestNetCommHookCategorization(TestCase):
    """NET rules for DDP comm-hook and TorchRec comm frames. C++ frame
    names use the demangled-with-signature form snapshots actually emit."""

    def test_ddp_all_reduce_bucket_is_net(self) -> None:
        frames = [
            Frame(
                name="c10d::Reducer::all_reduce_bucket(c10d::Reducer::Bucket&)",
                filename="<invalid>",
                line=0,
            ),
        ]
        self.assertEqual(AllocationType.from_frame_stack(frames), AllocationType.NET)

    def test_python_comm_hook_run_hook_is_net(self) -> None:
        frames = [
            Frame(
                name="c10d::PythonCommHook::runHook(c10d::GradBucket&)",
                filename="<invalid>",
                line=0,
            ),
        ]
        self.assertEqual(AllocationType.from_frame_stack(frames), AllocationType.NET)

    def test_ddp_comm_hooks_filename_is_net(self) -> None:
        frames = [
            Frame(
                name="default_hooks_allreduce",
                filename="torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py",
                line=20,
            ),
        ]
        self.assertEqual(AllocationType.from_frame_stack(frames), AllocationType.NET)

    def test_torchrec_comm_ops_filename_is_net(self) -> None:
        frames = [
            Frame(
                name="alltoall_pooled",
                filename="torchrec/distributed/comm_ops.py",
                line=300,
            ),
        ]
        self.assertEqual(AllocationType.from_frame_stack(frames), AllocationType.NET)

    def test_comm_hook_under_autograd_engine_is_net(self) -> None:
        # Locks NET-before-BACKWARD ordering: real comm-hook stacks
        # carry an autograd parent frame.
        frames = [
            Frame(
                name="_compress_hook",
                filename="torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py",
                line=71,
            ),
            Frame(
                name="c10d::PythonCommHook::runHook(c10d::GradBucket&)",
                filename="<invalid>",
                line=0,
            ),
            Frame(
                name="c10d::Reducer::all_reduce_bucket(c10d::Reducer::Bucket&)",
                filename="<invalid>",
                line=0,
            ),
            Frame(
                name="torch::autograd::Engine::thread_main("
                "std::shared_ptr<torch::autograd::GraphTask> const&)",
                filename="<invalid>",
                line=0,
            ),
        ]
        self.assertEqual(AllocationType.from_frame_stack(frames), AllocationType.NET)

    def test_torchrec_comm_ops_under_forward_is_net(self) -> None:
        # Locks NET-before-ACTIVATION ordering: TorchRec all-to-all
        # runs inside EBC.forward.
        frames = [
            Frame(
                name="alltoall_pooled",
                filename="torchrec/distributed/comm_ops.py",
                line=524,
            ),
            Frame(
                name="forward",
                filename="torchrec/distributed/embeddingbag.py",
                line=1124,
            ),
        ]
        self.assertEqual(AllocationType.from_frame_stack(frames), AllocationType.NET)

    def test_pure_autograd_engine_remains_backward(self) -> None:
        # Negative: bare autograd stack stays BACKWARD.
        frames = [
            Frame(
                name="torch::autograd::Engine::thread_main("
                "std::shared_ptr<torch::autograd::GraphTask> const&)",
                filename="<invalid>",
                line=0,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.BACKWARD
        )


class TestOptimizerCategorization(TestCase):
    """Tests for OPTIMIZER-domain rules: optimizer step / construction
    frames and Distributed Shampoo / TorchRec keyed-optimizer files."""

    def test_distributed_shampoo_step_is_optimizer(self) -> None:
        # Shampoo's step is a `step` frame inside the shampoo file.
        frames = [
            Frame(
                name="step",
                filename="distributed_shampoo/shampoo.py",
                line=500,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.OPTIMIZER
        )

    def test_vanilla_torch_optim_step_is_optimizer(self) -> None:
        # A generic torch.optim Optimizer.step() frame.
        frames = [
            Frame(
                name="step",
                filename="torch/optim/optimizer.py",
                line=180,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.OPTIMIZER
        )

    def test_torchrec_keyed_optimizer_step_is_optimizer(self) -> None:
        # TorchRec keyed optimizer step (matches via filename rule).
        frames = [
            Frame(
                name="some_internal_helper",
                filename="torchrec/optim/keyed.py",
                line=100,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.OPTIMIZER
        )

    def test_shampoo_preconditioner_construction_is_optimizer(self) -> None:
        # Shampoo preconditioner construction frame.
        frames = [
            Frame(
                name="_instantiate_shampoo_preconditioner_list",
                filename="distributed_shampoo/preconditioner.py",
                line=42,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.OPTIMIZER
        )

    def test_existing_init_group_still_works(self) -> None:
        # Regression check: the pre-existing rule for _init_group must
        # continue to map to OPTIMIZER even with the expanded name set.
        frames = [
            Frame(
                name="_init_group",
                filename="torch/optim/adam.py",
                line=60,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.OPTIMIZER
        )


class TestEmbeddingCategorization(TestCase):
    """Tests for EMBEDDING categorization rules."""

    def test_init_dmp_is_embedding(self) -> None:
        frames = [
            Frame(
                name="_init_dmp",
                filename="torchrec/distributed/model_parallel.py",
                line=300,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.EMBEDDING
        )

    def test_apply_2d_emb_sharding_is_embedding(self) -> None:
        frames = [
            Frame(
                name="apply_2d_emb_sharding",
                filename="torchrec/distributed/sharding/cw_sharding.py",
                line=120,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.EMBEDDING
        )

    def test_fbgemm_tbe_init_is_embedding(self) -> None:
        frames = [
            Frame(
                name="reset_uvm_cache_stats",
                filename=("fbgemm_gpu/split_table_batched_embeddings_ops_training.py"),
                line=900,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.EMBEDDING
        )

    def test_torchrec_ebc_sharding_is_embedding(self) -> None:
        frames = [
            Frame(
                name="ShardedEmbeddingBagCollection_init",
                filename="torchrec/distributed/embeddingbag.py",
                line=80,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.EMBEDDING
        )


class TestActivationFbgemmAndInductorCategorization(TestCase):
    """FBGEMM forward Ops, TBE PT2 / VBE lookup, and Inductor cache files."""

    def test_fbgemm_op_forward_is_activation(self) -> None:
        frames = [
            Frame(
                name="fbgemm_gpu::JaggedToPaddedDenseOp::forward",
                filename="fbgemm_gpu/jagged_tensor_ops.cpp",
                line=120,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.ACTIVATION
        )

    def test_tbe_pt2_lookup_function_is_activation(self) -> None:
        frames = [
            Frame(
                name="split_embedding_codegen_lookup_rowwise_adagrad_function_pt2",
                filename="fbgemm_gpu/split_embeddings_utils.py",
                line=300,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.ACTIVATION
        )

    def test_vbe_lookup_forward_is_activation(self) -> None:
        frames = [
            Frame(
                name="SplitVBELookupFunction_v1::forward",
                filename="fbgemm_gpu/vbe_lookup.cpp",
                line=80,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.ACTIVATION
        )

    def test_inductor_cache_file_is_activation(self) -> None:
        frames = [
            Frame(
                name="call",
                filename="/tmp/torchinductor_user/ab/cabcdef12.py",
                line=5,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.ACTIVATION
        )

    def test_non_inductor_cache_file_does_not_match(self) -> None:
        # Negative: non-cache .py falls through to UNKNOWN.
        frames = [
            Frame(
                name="some_helper",
                filename="/var/tmp/foo.py",
                line=10,
            ),
        ]
        self.assertEqual(
            AllocationType.from_frame_stack(frames), AllocationType.UNKNOWN
        )


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
