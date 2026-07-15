"""Eval Suite: unified evaluation control plane.

Families:
    ppl         PPL / scalar metric evaluation
    benchmark   Benchmark task evaluation (deploy + client Hope jobs)

Quick start::

    from src.eval.suites import load_suite
    from src.eval.registry import build_run, create_runner

    suite = load_suite("configs/eval_suites/ppl/smoke.yaml")
    run = build_run(
        suite_path="configs/eval_suites/ppl/smoke.yaml",
        model_path="/path/to/model",
        model_tag="baseline",
        output_dir="output/olmo3-32b/d1_math",
    )
    runner = create_runner(run)
    result = runner.execute()
"""
