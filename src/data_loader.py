from pathlib import Path
import networkx as nx


def read_giv(path):
    with open(path, encoding="utf-8-sig") as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                yield [x.strip() for x in line.split(";")]


def load_dataset(folder):
    folder = Path(folder)

    stops = {
        int(sid): {
            "short_name": short,
            "long_name": name,
            "x": float(x),
            "y": float(y)
        }
        for sid, short, name, x, y in read_giv(folder / "Stop.giv")
    }

    edges = {
        int(eid): {
            "from_stop": int(u),
            "to_stop": int(v),
            "length": float(length),
            "lower_bound": float(lower),
            "upper_bound": float(upper)
        }
        for eid, u, v, length, lower, upper in read_giv(folder / "Edge.giv")
    }

    demand = {
        (int(o), int(d)): float(passengers)
        for o, d, passengers in read_giv(folder / "OD.giv")
    }

    graph = nx.Graph()
    graph.add_nodes_from((sid, info) for sid, info in stops.items())

    for eid, edge in edges.items():
        graph.add_edge(
            edge["from_stop"],
            edge["to_stop"],
            edge_id=eid,
            weight=edge["lower_bound"],
            length=edge["length"]
        )

    pool = None
    if (folder / "Pool.giv").exists():
        pool = {}
        for line_id, order, edge_id in read_giv(folder / "Pool.giv"):
            pool.setdefault(int(line_id), []).append(
                (int(order), int(edge_id))
            )

        pool = {
            lid: [eid for _, eid in sorted(items)]
            for lid, items in pool.items()
        }

    return {
        "stops": stops,
        "edges": edges,
        "demand": demand,
        "graph": graph,
        "pool": pool
    }
