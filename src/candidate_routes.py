from itertools import islice
import networkx as nx


def enumerate_candidates(data, max_time=40) -> dict:
    """Enumerate simple routes up to max_time minutes, merging reversals.

    Keep subpaths and sort stop sequences for stable route IDs.
    Positive edge times let us stop extending routes at the time limit.
    """
    graph = data["graph"]
    if any(info["weight"] <= 0 for _, _, info in graph.edges(data=True)):
        raise ValueError("Route enumeration requires positive edge travel times")
    routes = {}

    def extend(path, elapsed):
        if len(path) > 1 and path[0] < path[-1]:
            routes[tuple(path)] = make_route(path, graph)
        for stop in sorted(graph[path[-1]]):
            travel_time = elapsed + graph[path[-1]][stop]["weight"]
            if stop not in path and travel_time <= max_time:
                extend(path + [stop], travel_time)

    for stop in sorted(graph):
        extend([stop], 0.0)
    return {i: routes[path] for i, path in enumerate(sorted(routes), 1)}


def route_from_edges(edge_ids, edges):
    first = edges[edge_ids[0]]
    endpoints = (first["from_stop"], first["to_stop"])

    for start, end in (endpoints, endpoints[::-1]):
        stops = [start, end]
        for eid in edge_ids[1:]:
            edge = edges[eid]
            u, v = edge["from_stop"], edge["to_stop"]
            if stops[-1] == u:
                stops.append(v)
            elif stops[-1] == v:
                stops.append(u)
            else:
                break
        else:
            return stops
    raise ValueError(f"Edges do not form a continuous route: {edge_ids}")


def make_route(stops, graph):
    pairs = list(zip(stops, stops[1:]))
    return {
        "stops": list(stops),
        "edge_ids": [graph[u][v]["edge_id"] for u, v in pairs],
        "travel_time": sum(graph[u][v]["weight"] for u, v in pairs),
        "length": sum(graph[u][v]["length"] for u, v in pairs),
    }


def filter_subpaths(routes):
    filtered = {}

    for rid, route in routes.items():
        short = route["stops"]
        is_contained = False

        for other_id, other in routes.items():
            if rid == other_id:
                continue

            long = other["stops"]

            if len(short) >= len(long):
                continue

            for i in range(len(long) - len(short) + 1):
                segment = long[i:i + len(short)]

                if segment == short or segment == short[::-1]:
                    is_contained = True
                    break

            if is_contained:
                break

        if not is_contained:
            filtered[rid] = route

    return filtered


def get_candidates(data, k, max_time): 
    graph = data["graph"]
    routes = {}

    if data["pool"] is not None:
        for lid, edge_ids in data["pool"].items():
            stops = route_from_edges(edge_ids, data["edges"])
            route = make_route(stops, graph)
            if route["travel_time"] <= max_time:
                routes[lid] = route
                
            
        return routes

    seen = set()
    for (origin, destination), demand in data["demand"].items():
        if demand <= 0 or origin == destination:
            continue
        try:
            paths = nx.shortest_simple_paths(
                graph, origin, destination, weight="weight"
            )
            for path in islice(paths, k):
                route = make_route(path, graph)
                if route["travel_time"] > max_time:
                    break
                key = min(tuple(path), tuple(reversed(path)))
                if key not in seen:
                    seen.add(key)
                    routes[len(routes) + 1] = route
        except nx.NetworkXNoPath:
            continue
    
    return routes

