"""Compare both heuristics on Mandl's network."""

import argparse
import contextlib
import io
import json
from pathlib import Path
from time import perf_counter

from src.candidate_routes import enumerate_candidates
from src.candidate_routes import get_candidates
from src.column_generation import solve_column_generation
from src.data_loader import load_dataset
from src.exact_method import solve_exact
from src.heuristic_method import solve_heuristic


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    data = load_dataset(Path(__file__).parent / "dataset/mandl/basis")
    pools = {"k3": get_candidates(data, 3, 40), "all40": enumerate_candidates(data, 40)}
    config = {
        "alpha": 1.5,
        "scenarios": [[5, 150], [3, 100]],
        "seeds": [42, *range(9)],
        "restarts": 30,
        "top_starts": 8,
        "max_local_steps": 20,
        "cg_seed": 42,
        "batch_size": 20,
        "tolerance": 1e-7,
        "milp_time_limit": 300,
        "k": 3,
        "max_route_time": 40,
    }
    (output / "config.json").write_text(json.dumps(config) + "\n")
    for name, pool in pools.items():
        (output / f"pool-{name}.json").write_text(json.dumps(pool) + "\n")
        for p, budget in config["scenarios"]:
            methods = [("greedy_local", seed) for seed in config["seeds"]]
            for method, seed in methods + [("cg", 42), ("reference", None)]:
                start = perf_counter()
                with contextlib.redirect_stdout(io.StringIO()):
                    if method == "greedy_local":
                        result = solve_heuristic(
                            data, pool, p, budget, restarts=30, seed=seed
                        )
                    elif method == "cg":
                        result = solve_column_generation(data, pool, p, budget)
                    else:
                        result = solve_exact(data, pool, p, budget)
                result["runtime"] = perf_counter() - start
                record = {
                    "pool": name,
                    "p": p,
                    "B": budget,
                    "method": method,
                    "seed": seed,
                    **result,
                }
                if result.get("feasible"):
                    record["selected_routes"] = [
                        pool[i]["stops"] for i in result["selected_lines"]
                    ]
                with (output / "runs.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
    print(f"Saved 48 runs to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
