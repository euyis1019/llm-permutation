"""External benchmark adapters.

Some benchmarks are evaluated by third-party "scaffolding" frameworks (e.g.
evalplus for HumanEval+/MBPP+) rather than by our protocol-driven client_runner.
Such a benchmark is a *black box*: the framework owns prompt construction,
generation, and scoring (sandbox pass@k), so it cannot be expressed via the
``prompt_builder_id`` / ``scorer_id`` protocol contract.

This package keeps those frameworks out of the protocol layer.  An external
bench runs as a single in-process GPU job (the framework loads vLLM itself and
relies on vLLM continuous batching to keep the GPU saturated — there is no
deploy/client split and no concurrency knob).  Its raw result is normalized here
to the same summary schema protocol benches produce, so the H1 report can
aggregate both uniformly.

Design notes & decisions:
  experiments/10_h1_convergence/01_suite_definition/test_for_integration/
  DESIGN_external_eval_integration.md

Offline-node engineering (no DNS on GPU nodes) that the worker script encodes:
  经验总结-MLP-Hope-踩坑.md §八
"""
