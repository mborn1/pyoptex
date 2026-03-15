#!/usr/bin/env python3
"""
One-time script to run all examples and capture @@CHECKPOINT@@ output
into JSON reference files under tests/reference_data/.

Usage (from workspace root):
    venv/bin/python tests/_capture_references.py
"""

import json
import os
import subprocess
import sys
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).parent.parent / "examples"
REFERENCE_ROOT = Path(__file__).parent / "references"

EXAMPLE_MAP = {
    # (script_path_relative_to_examples, reference_key)
    "analysis/simple_model.py": "analysis/simple_model",
    "analysis/simple_model_mixedlm.py": "analysis/simple_model_mixedlm",
    "analysis/drop_pvalue.py": "analysis/drop_pvalue",
    "analysis/drop_pvalue_outliers.py": "analysis/drop_pvalue_outliers",
    "analysis/drop_pvalue_strong.py": "analysis/drop_pvalue_strong",
    "analysis/drop_pvalue_strong_cat.py": "analysis/drop_pvalue_strong_cat",
    "analysis/sams/sams_generic.py": "analysis/sams_generic",
    "analysis/sams/sams_generic_all.py": "analysis/sams_generic_all",
    "analysis/sams/sams_partial_rsm.py": "analysis/sams_partial_rsm",
    "analysis/sams/sams_partial_rsm_cluster.py": "analysis/sams_partial_rsm_cluster",
    "doe/quickstart/example_randomized_fs.py": "doe_quickstart/randomized_fs",
    "doe/quickstart/example_randomized_sp.py": "doe_quickstart/randomized_sp",
    "doe/quickstart/example_splitplot_fs.py": "doe_quickstart/splitplot_fs",
    "doe/quickstart/example_splitplot_sp.py": "doe_quickstart/splitplot_sp",
    "doe/quickstart/example_splitplot_multiprocessing.py": "doe_quickstart/splitplot_multiprocessing",
    "doe/quickstart/example_strip_plot_fs.py": "doe_quickstart/strip_plot_fs",
    "doe/quickstart/example_cost_optimal_codex.py": "doe_quickstart/cost_optimal_codex",
    "doe/quickstart/example_cost_optimal_codex_mp.py": "doe_quickstart/cost_optimal_codex_mp",
    "doe/cost_optimal/codex/example_codex.py": "doe_cost_optimal/codex",
    "doe/cost_optimal/codex/example_approx_omars.py": "doe_cost_optimal/approx_omars",
    "doe/cost_optimal/codex/example_asymmetric.py": "doe_cost_optimal/asymmetric",
    "doe/cost_optimal/codex/example_micro_pharma.py": "doe_cost_optimal/micro_pharma",
    "doe/cost_optimal/codex/example_mixture.py": "doe_cost_optimal/mixture",
    "doe/cost_optimal/codex/example_scaled.py": "doe_cost_optimal/scaled",
    "doe/fixed_structure/example_approx_omars.py": "doe_fixed_structure/approx_omars",
    "doe/fixed_structure/example_mixture.py": "doe_fixed_structure/mixture",
    "doe/fixed_structure/example_strip_plot.py": "doe_fixed_structure/strip_plot",
    "doe/fixed_structure/splitk_plot/example_splitk_augment.py": "doe_fixed_structure/splitk_augment",
    "doe/fixed_structure/splitk_plot/example_splitk_augment_split.py": "doe_fixed_structure/splitk_augment_split",
    "doe/fixed_structure/splitk_plot/example_splitk_fixed_factor.py": "doe_fixed_structure/splitk_fixed_factor",
    "doe/fixed_structure/splitk_plot/example_splitk_plot.py": "doe_fixed_structure/splitk_plot",
}


def parse_checkpoints(stdout: str) -> dict:
    checkpoints = {}
    for line in stdout.splitlines():
        if line.startswith("@@CHECKPOINT@@"):
            payload = line[len("@@CHECKPOINT@@") :]
            try:
                obj = json.loads(payload)
                checkpoints[obj["name"]] = obj["value"]
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  WARNING: failed to parse checkpoint line: {line!r}: {e}", file=sys.stderr)
    return checkpoints


def run_example(script_rel: str) -> dict:
    script_path = EXAMPLES_ROOT / script_rel
    env = os.environ.copy()
    # Add examples parent to PYTHONPATH so `from examples._log_checkpoint import ...` works
    workspace_root = str(EXAMPLES_ROOT.parent)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{workspace_root}:{existing}" if existing else workspace_root

    print(f"  Running {script_rel} ...", end=" ", flush=True)
    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        cwd=str(script_path.parent),
    )
    if result.returncode != 0:
        print(f"FAILED (exit {result.returncode})")
        print("  STDERR:", result.stderr[:2000])
        return {}
    checkpoints = parse_checkpoints(result.stdout)
    print(f"OK ({len(checkpoints)} checkpoints)")
    return checkpoints


def main():
    REFERENCE_ROOT.mkdir(parents=True, exist_ok=True)

    failed = []
    for script_rel, ref_key in EXAMPLE_MAP.items():
        print(f"\n[{ref_key}]")
        checkpoints = run_example(script_rel)
        if not checkpoints:
            failed.append(script_rel)
            continue

        ref_path = REFERENCE_ROOT / f"{ref_key}.json"
        ref_path.parent.mkdir(parents=True, exist_ok=True)
        with open(ref_path, "w") as f:
            json.dump(checkpoints, f, indent=2)
        print(f"  Saved -> {ref_path.relative_to(Path.cwd())}")

    print("\n" + "=" * 60)
    if failed:
        print(f"FAILED ({len(failed)}):")
        for s in failed:
            print(f"  {s}")
        sys.exit(1)
    else:
        print(f"All {len(EXAMPLE_MAP)} examples captured successfully.")


if __name__ == "__main__":
    main()
