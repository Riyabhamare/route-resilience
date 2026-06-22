import cv2
import numpy as np
import networkx as nx
import sknw
from skimage.morphology import skeletonize


def clean_mask(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    return cleaned


def to_skeleton(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    skel   = skeletonize(binary)
    return skel.astype(np.uint8)


def build_graph(skel: np.ndarray):
    return sknw.build_sknw(skel)


def compute_edge_betweenness(graph) -> dict:
    """
    Compute edge betweenness centrality for all edges.

    Uses length_px as the edge weight so that shorter pixel paths are
    preferred when finding shortest paths (realistic road cost).
    normalized=True keeps scores in [0, 1] and comparable across graphs.

    Returns
    -------
    dict  {(u, v): float}  — centrality score per edge
    """
    # sknw stores edge length in 'weight'; rename to length_px for clarity
    # NetworkX edge_betweenness needs the weight key to exist on edges
    for u, v, data in graph.edges(data=True):
        data['length_px'] = float(data.get('weight', 1.0))

    centrality = nx.edge_betweenness_centrality(
        graph,
        weight='length_px',
        normalized=True,
    )
    return centrality


def extract_graph_data(graph) -> dict:
    """
    Convert a sknw NetworkX graph to a serialisable dict.

    Adds per-edge betweenness centrality scores and a top-10
    critical_edges list (highest centrality first).

    Returns
    -------
    {
        "nodes": {nid: {"pixel_coord": (row, col)}},
        "edges": [{"from", "to", "pixel_path", "length_px",
                   "betweenness_centrality"}],
        "critical_edges": [
            {"from", "to", "betweenness_centrality", "length_px"}, ...
        ]   # top-10 by centrality — bridges & narrow underpasses
    }
    """
    # ── 1. Edge betweenness centrality ──────────────────────────────────────
    centrality = compute_edge_betweenness(graph)

    # ── 2. Nodes ─────────────────────────────────────────────────────────────
    nodes = {
        nid: {"pixel_coord": tuple(data["o"].astype(int))}
        for nid, data in graph.nodes(data=True)
    }

    # ── 3. Edges (with centrality stored as attribute) ───────────────────────
    edges = []
    for u, v, data in graph.edges(data=True):
        # centrality dict keys may be (u,v) or (v,u) — check both
        score = centrality.get((u, v), centrality.get((v, u), 0.0))

        edge_record = {
            "from":                   u,
            "to":                     v,
            "pixel_path":             data["pts"].tolist(),
            "length_px":              float(data.get('weight', 1.0)),
            "betweenness_centrality": round(score, 6),
        }
        edges.append(edge_record)

    # ── 4. Critical edges — top-10 by betweenness ───────────────────────────
    sorted_edges = sorted(
        edges,
        key=lambda e: e["betweenness_centrality"],
        reverse=True,
    )
    critical_edges = [
        {
            "from":                   e["from"],
            "to":                     e["to"],
            "betweenness_centrality": e["betweenness_centrality"],
            "length_px":              e["length_px"],
        }
        for e in sorted_edges[:10]
    ]

    return {
        "nodes":          nodes,
        "edges":          edges,
        "critical_edges": critical_edges,
    }


def mask_to_graph(mask: np.ndarray) -> dict:
    cleaned = clean_mask(mask)
    skel    = to_skeleton(cleaned)
    graph   = build_graph(skel)
    return extract_graph_data(graph)