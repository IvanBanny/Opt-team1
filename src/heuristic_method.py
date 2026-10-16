# Randomized greedy construction + one-line-swap local search


from random import Random
from time import perf_counter
import networkx as nx


def _prepare(data, candidates, alpha):
    if alpha < 1:
        raise ValueError("alpha must be at least 1")
    graph = data["graph"]
    if not nx.is_connected(graph):
        raise ValueError("Infrastructure graph is disconnected")

    demand = {
        (s, t): amount
        for (s, t), amount in data["demand"].items()
        if amount > 0 and s != t
    }
    shortest = dict(nx.all_pairs_dijkstra_path_length(graph, weight="weight"))
    covered_by_line = {}
    costs = {}
    route_stops = {}

    for line_id, route_info in candidates.items():
        route = route_info["stops"]
        route_stops[line_id] = set(route)
        costs[line_id] = float(route_info["travel_time"])
        cumulative = [0.0]
        for u, v in zip(route, route[1:]):
            cumulative.append(cumulative[-1] + graph[u][v]["weight"])
        positions = {stop: idx for idx, stop in enumerate(route)}

        covered_by_line[line_id] = {
            (s, t)
            for s, t in demand
            if s in positions and t in positions
            and abs(cumulative[positions[s]] - cumulative[positions[t]])
            <= alpha * shortest[s][t] + 1e-9
        }

    return demand, covered_by_line, costs, route_stops


def _components(data, candidates, selected):
    g = nx.Graph()
    g.add_nodes_from(data["graph"].nodes)
    for line_id in selected:
        route = candidates[line_id]["stops"]
        g.add_edges_from(zip(route, route[1:]))
    return list(nx.connected_components(g))


def solve_heuristic(data, candidates, p, B, alpha=1.5,
                    restarts=120, top_starts=8, seed=42,
                    max_local_steps=20):
    
    start_time = perf_counter()
    if not candidates:
        raise ValueError("Candidate route pool is empty")
    if p < 1 or B <= 0:
        raise ValueError("p must be >= 1 and B must be positive")

    rng = Random(seed)
    demand, covers, costs, stops_on_route = _prepare(data, candidates, alpha)
    line_ids = list(candidates)
    num_stops = data["graph"].number_of_nodes()
    total_demand = sum(demand.values())

    def coverage(selected):
        od_covered = set()
        for line_id in selected:
            od_covered.update(covers[line_id])
        return sum(demand[od] for od in od_covered)

    def feasible(selected):
        return (len(selected) <= p
                and sum(costs[l] for l in selected) <= B + 1e-9
                and len(_components(data, candidates, selected)) == 1)

    def greedy_construct(restart_idx):
        selected = []
        covered = set()
        spent = 0.0
        # Try different balances between connecting stops and covering demand.
        connection_weight = (1, 2, 4, 8, 16)[restart_idx % 5]
        rcl_size = (1, 2, 3, 5)[(restart_idx // 5) % 4]

        while len(selected) < p:
            components = _components(data, candidates, selected)
            comp_of = {stop: idx for idx, comp in enumerate(components)
                       for stop in comp}
            connected = len(components) == 1
            choices = []
            for line_id in line_ids:
                if line_id in selected or spent + costs[line_id] > B + 1e-9:
                    continue
                gain = sum(demand[od] for od in covers[line_id] - covered)
                # A route links all components containing any of its stops.
                connected_components = {comp_of[s] for s in stops_on_route[line_id]}
                component_reduction = len(connected_components) - 1
                if connected and gain == 0:
                    continue
                if gain == 0 and component_reduction == 0:
                    continue
                score = ((gain / total_demand if total_demand else 0.0)
                         + connection_weight * component_reduction / max(1, num_stops - 1))
                # A small preference for lower-cost lines, without making
                # cost more important than connectivity and coverage.
                score /= 1.0 + 0.1 * costs[line_id] / B
                choices.append((score, line_id))

            if not choices:
                break
            choices.sort(reverse=True)
            _, chosen = rng.choice(choices[:min(rcl_size, len(choices))])
            selected.append(chosen)
            covered.update(covers[chosen])
            spent += costs[chosen]
        return selected

    def local_search(initial):
        selected = list(initial)
        current = coverage(selected)
        for _ in range(max_local_steps):
            best_score = current
            best_neighbor = None
            selected_set = set(selected)
            available = [l for l in line_ids if l not in selected_set]

            # Add a line if a free slot and enough budget remain.
            if len(selected) < p:
                for new in available:
                    neighbor = selected + [new]
                    if sum(costs[l] for l in neighbor) <= B + 1e-9:
                        score = coverage(neighbor)
                        if score > best_score and feasible(neighbor):
                            best_score, best_neighbor = score, neighbor

            # Replace one selected line with one unselected line.
            for old in selected:
                for new in available:
                    neighbor = [new if l == old else l for l in selected]
                    if sum(costs[l] for l in neighbor) > B + 1e-9:
                        continue
                    score = coverage(neighbor)
                    if score > best_score and feasible(neighbor):
                        best_score, best_neighbor = score, neighbor

            if best_neighbor is None:
                break
            selected, current = best_neighbor, best_score
        return selected

    feasible_starts = {}
    for idx in range(restarts):
        selected = greedy_construct(idx)
        if feasible(selected):
            key = frozenset(selected)
            feasible_starts[key] = selected

    if not feasible_starts:
        return {
            "feasible": False,
            "message": "No feasible solution found by randomized greedy; try more restarts or a connectivity-aware repair step.",
            "feasible_starts": 0,
            "runtime": perf_counter() - start_time,
        }

    initial_solutions = sorted(feasible_starts.values(), key=coverage, reverse=True)
    best = initial_solutions[0]
    best_score = coverage(best)
    greedy_best_score = best_score

    # Compare the best feasible greedy solution with local-search results.
    print("\n--- Greedy Only ---")
    print("Restarts:", restarts)
    print("Unique feasible starts:", len(feasible_starts))
    print("Selected lines:", best)
    print("Covered demand:", greedy_best_score)
    print("Coverage (%):", round(100 * greedy_best_score / total_demand, 2)
          if total_demand else 0.0)

    print("\n--- Local Search Progress ---")
    for i, initial in enumerate(initial_solutions[:top_starts], 1):
        before = coverage(initial)
        improved = local_search(initial)
        candidate_score = coverage(improved)
        print(f"Start {i}: {before:.0f} -> {candidate_score:.0f} "
              f"(improvement: {candidate_score - before:+.0f})")
        if candidate_score > best_score:
            best, best_score = improved, candidate_score

    print("\n--- Greedy vs Local Search ---")
    print("Best greedy covered demand:", greedy_best_score)
    print("Best after local search:", best_score)
    print("Overall improvement:", best_score - greedy_best_score)
    print("Final selected lines:", best)

    result = {
        "feasible": True,
        "selected_lines": best,
        "objective": float(best_score),
        "total_cost": sum(costs[l] for l in best),
        "coverage_pct": 100 * best_score / total_demand if total_demand else 0.0,
        "runtime": perf_counter() - start_time,
        "feasible_starts": len(feasible_starts),
        "restarts": restarts,
    }
    return result
