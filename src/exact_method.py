from itertools import combinations
from time import perf_counter

import networkx as nx
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

def check_connectivity(data, candidates, selected_lines):
    graph = nx.Graph()
    graph.add_nodes_from(data["graph"].nodes())

    for l in selected_lines:
        route = candidates[l]["stops"]
        graph.add_edges_from(zip(route, route[1:]))

    return nx.is_connected(graph)


def solve_exact(data, candidates, p, B, alpha=1.5,
                time_limit=300, return_model=False):
    graph = data["graph"]

    if not nx.is_connected(graph):
        raise ValueError("Infrastructure graph is disconnected.")

    if alpha < 1:
        raise ValueError("alpha must be at least 1.")

    # OD pairs with demand
    demand = {
        pair: value
        for pair, value in data["demand"].items()
        if value > 0 and pair[0] != pair[1]
    }

    line_ids = list(candidates.keys())
    od_pairs = list(demand.keys())
    stops = list(graph.nodes())

    n_lines = len(line_ids)
    n_od = len(od_pairs)
    n_vars = n_lines + n_od

    if n_lines == 0:
        raise ValueError("Candidate route pool is empty.")


    line_index = {
        l: i for i, l in enumerate(line_ids)
    }

    od_index = {
        pair: n_lines + i
        for i, pair in enumerate(od_pairs)
    }

    # Shortest travel times
    shortest_times = dict(
        nx.all_pairs_dijkstra_path_length(
            graph, weight="weight"
        )
    )

    # Line costs
    costs = {
        l: candidates[l]["travel_time"]
        for l in line_ids
    }

    # Direct connections within the detour limit
    acceptable = {}

    for l in line_ids:
        route = candidates[l]["stops"]

        cumulative = [0.0]

        for u, v in zip(route, route[1:]):
            cumulative.append(
                cumulative[-1] + graph[u][v]["weight"]
            )

        positions = {
            stop: i for i, stop in enumerate(route)
        }

        for s, t in od_pairs:
            if s not in positions or t not in positions:
                continue

            direct_time = abs(
                cumulative[positions[s]]
                - cumulative[positions[t]]
            )

            if direct_time <= alpha * shortest_times[s][t] + 1e-9:
                acceptable[l, s, t] = 1

    # Maximize covered demand
    c = np.zeros(n_vars)

    for (s, t), passengers in demand.items():
        c[od_index[s, t]] = -passengers

    integrality = np.zeros(n_vars, dtype=int)
    integrality[:n_lines] = 1

    bounds = Bounds(
        lb=np.zeros(n_vars),
        ub=np.ones(n_vars)
    )

    # Constraints
    rows = []
    cols = []
    values = []
    lower = []
    upper = []

    def add_constraint(coefficients, lb=-np.inf, ub=np.inf):
        row_id = len(lower)

        for col_id, value in coefficients.items():
            if value != 0:
                rows.append(row_id)
                cols.append(col_id)
                values.append(value)

        lower.append(lb)
        upper.append(ub)

    for s, t in od_pairs:
        coeff = {od_index[s, t]: 1.0}

        for l in line_ids:
            if acceptable.get((l, s, t), 0) == 1:
                coeff[line_index[l]] = -1.0

        add_constraint(coeff, ub=0.0)

    # Line limit
    add_constraint(
        {line_index[l]: 1.0 for l in line_ids},
        ub=p
    )

    # Effort budget
    add_constraint(
        {line_index[l]: costs[l] for l in line_ids},
        ub=B
    )

    # Connectivity cuts
    root = stops[0]
    other_stops = [v for v in stops if v != root]

    route_edges = {
        l: list(zip(
            candidates[l]["stops"],
            candidates[l]["stops"][1:]
        ))
        for l in line_ids
    }

    cut_count = 0

    for size in range(1, len(other_stops) + 1):
        for subset in combinations(other_stops, size):
            U = set(subset)

            crossing = {}

            for l in line_ids:
                crosses = any(
                    (u in U) != (v in U)
                    for u, v in route_edges[l]
                )

                if crosses:
                    crossing[line_index[l]] = 1.0

            add_constraint(crossing, lb=1.0)
            cut_count += 1

    A = coo_matrix(
        (values, (rows, cols)),
        shape=(len(lower), n_vars)
    ).tocsr()

    constraints = LinearConstraint(
        A,
        lb=np.array(lower),
        ub=np.array(upper)
    )

    if return_model:
        return c, constraints

    print(f"Candidate lines: {n_lines}")
    print(f"Positive-demand OD pairs: {n_od}")
    print(f"Connectivity constraints: {cut_count}")
    print(f"Total variables: {n_vars}")
    print(f"Total constraints: {len(lower)}")

    # Solve
    start = perf_counter()

    result = milp(
        c=c,
        integrality=integrality,
        bounds=bounds,
        constraints=constraints,
        options={
            "disp": False,
            "mip_rel_gap": 0.0,
            "time_limit": time_limit,
        }
    )

    runtime = perf_counter() - start
    bound = getattr(result, "mip_dual_bound", None)
    upper_bound = (-float(bound)
                   if bound is not None and np.isfinite(bound) else None)

    print("\n--- Exact method results ---")
    print("Solver message:", result.message)
    print("Runtime (seconds):", round(runtime, 3))

    if result.x is None:
        print("No feasible solution was returned.")
        return {
            "feasible": False,
            "upper_bound": upper_bound,
            "gap": getattr(result, "mip_gap", None),
            "status": result.status,
            "message": result.message,
            "runtime": runtime,
        }

    selected = [
        l for l in line_ids
        if result.x[line_index[l]] > 0.5
    ]

    print(
        "Selected network connected:",
        check_connectivity(data, candidates, selected)
    )

    covered_demand = sum(
        passengers * result.x[od_index[s, t]]
        for (s, t), passengers in demand.items()
    )

    total_demand = sum(demand.values())
    total_cost = sum(costs[l] for l in selected)

    print("Selected lines:", selected)
    print("Covered passenger demand:", covered_demand)
    print("Total passenger demand:", total_demand)

    print(
        "Direct coverage (%):",
        round(100 * covered_demand / total_demand, 2)
        if total_demand > 0 else 0
    )

    print("Operating cost:", total_cost)
    print("Optimality gap:", getattr(result, "mip_gap", None))
    print("Search nodes:", getattr(result, "mip_node_count", None))

    for l in selected:
        print(
            f"Line {l}: {candidates[l]['stops']}"
            f" (cost={costs[l]:.2f})"
        )

    return {
        "feasible": True,
        "upper_bound": upper_bound,
        "selected_lines": selected,
        "objective": covered_demand,
        "total_cost": total_cost,
        "runtime": runtime,
        "gap": getattr(result, "mip_gap", None),
        "status": result.status,
        "message": result.message,
    }

