"""Column generation and integer route selection."""

import contextlib
import io
import time

import numpy as np
from scipy import optimize

from src.exact_method import solve_exact
from src.heuristic_method import solve_heuristic


def solve_column_generation(
    data,
    candidates,
    p,
    B,
    alpha=1.5,
    batch_size=20,
    time_limit=300,
    tolerance=1e-7,
):
    """Solve the LP by column generation, then select integer routes."""
    start = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        initial = solve_heuristic(
            data, candidates, p, B, alpha, restarts=30, top_starts=0, seed=42
        )
    if not initial["feasible"]:
        return {
            **initial,
            "termination": "initialization_failed",
            "lp_certified": False,
            "full_lp_bound": None,
        }
    times = {"initialization": time.perf_counter() - start, "lp": 0.0, "pricing": 0.0}
    stage = time.perf_counter()
    c, constraints = solve_exact(data, candidates, p, B, alpha, return_model=True)
    # flip >= connectivity rows so linprog can use them
    signs = np.where(np.isfinite(constraints.ub), 1.0, -1.0)
    A = constraints.A.multiply(signs[:, None]).tocsc()
    b = np.where(signs > 0, constraints.ub, -constraints.lb)
    times["model"] = time.perf_counter() - stage
    line_ids = list(candidates)
    generated = list(initial["selected_lines"])
    index = {line_id: i for i, line_id in enumerate(line_ids)}
    trace = []
    bound = None
    termination = "lp_failed"
    while True:
        columns = [index[line_id] for line_id in generated]
        columns += list(range(len(candidates), len(c)))
        stage = time.perf_counter()
        lp = optimize.linprog(
            c[columns],
            A_ub=A[:, columns],
            b_ub=b,
            bounds=(0, 1),
            method="highs",
            options={
                "dual_feasibility_tolerance": tolerance,
                "primal_feasibility_tolerance": tolerance,
            },
        )
        times["lp"] += time.perf_counter() - stage
        if not lp.success:
            break
        stage = time.perf_counter()
        dual = np.minimum(lp.ineqlin.marginals, 0.0)
        reduced = c - np.asarray(A.T @ dual).ravel()
        omitted = [i for i, line_id in enumerate(line_ids) if line_id not in generated]
        improving = [
            i
            for i in sorted(omitted, key=lambda i: reduced[i])
            if reduced[i] < -tolerance
        ]
        added = [line_ids[i] for i in improving[:batch_size]]
        trace.append(
            {
                "columns": len(generated),
                "lp_objective": -float(lp.fun),
                "min_omitted_reduced_cost": float(min(reduced[omitted]))
                if omitted
                else None,
                "added": added,
            }
        )
        times["pricing"] += time.perf_counter() - stage
        if not improving:
            # account for [0, 1] bounds, including omitted routes
            bound = -float(b @ dual + np.minimum(reduced, 0).sum())
            termination = "lp_optimal"
            break
        generated.extend(added)

    stage = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        integer = solve_exact(
            data,
            {i: candidates[i] for i in generated},
            p,
            B,
            alpha,
            time_limit=time_limit,
        )
    times["milp"] = time.perf_counter() - stage
    best = (
        integer
        if integer.get("objective", -np.inf) >= initial["objective"]
        else initial
    )
    return {
        **best,
        "initialization": initial,
        "restricted_milp": integer,
        "generated_lines": generated,
        "trace": trace,
        "timings": times,
        "termination": termination,
        "lp_status": lp.status,
        "lp_message": lp.message,
        "lp_certified": bound is not None,
        "full_lp_bound": bound,
        "full_lp_objective": -float(lp.fun) if bound is not None else None,
        "integer_certified": bool(
            bound is not None and bound - best["objective"] <= tolerance
        ),
        "runtime": time.perf_counter() - start,
    }
