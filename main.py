from pathlib import Path
from time import perf_counter
from src.data_loader import load_dataset
from src.candidate_routes import get_candidates
from src.exact_method import solve_exact
from src.heuristic_method import solve_heuristic

root = Path(__file__).resolve().parent

# Select dataset
data = load_dataset(root / "dataset/mandl/basis")
# data = load_dataset(root / "dataset/toy/basis")

# Candidate generation parameters
k = 3
max_time = 40

candidates = get_candidates(data, k=k, max_time=max_time)


# Experiments
alpha = 1.5
scenarios = [
    ("Baseline", 5, 150),
    ("Strict constraints", 3, 100)
]


for name, p, B in scenarios:

    print(f"\n{'=' * 60}")
    print(f"EXPERIMENT: {name}")
    print(f"p = {p}, B = {B}, alpha = {alpha}")
    print(f"{'=' * 60}")

    # Exact method
    exact_start = perf_counter()

    exact_results = solve_exact(
        data=data,
        candidates=candidates,
        p=p,
        B=B,
        alpha=alpha
    )

    exact_time = perf_counter() - exact_start

    # Heuristic method
    heuristic_start = perf_counter()

    heuristic_results = solve_heuristic(
        data=data,
        candidates=candidates,
        p=p,
        B=B,
        alpha=alpha,
        restarts=30,
        seed=42
    )

    heuristic_time = perf_counter() - heuristic_start

    # Summary
    print(f"\n--- SUMMARY: {name} ---")

    print("Exact status:", exact_results.get("message"))
    print("Exact covered demand:", exact_results.get("objective"))
    print("Exact total runtime:", round(exact_time, 3), "seconds")

    print("Heuristic feasible:", heuristic_results.get("feasible"))
    print("Heuristic covered demand:", heuristic_results.get("objective"))
    print("Heuristic total runtime:", round(heuristic_time, 3), "seconds")

    if (
        exact_results.get("status") == 0
        and heuristic_results.get("feasible")
    ):
        exact_obj = exact_results["objective"]
        heuristic_obj = heuristic_results["objective"]

        if exact_obj > 0:
            gap = 100 * (exact_obj - heuristic_obj) / exact_obj
        else:
            gap = 0.0 if heuristic_obj == 0 else None

        print(
            "Heuristic optimality gap (%):",
            round(gap, 2) if gap is not None else "N/A"
        )
    else:
        print("Heuristic optimality gap (%): N/A")