"""
healing.py — MST Healing with Angle Constraint
================================================
Reconnects disconnected components in a road skeleton graph using a
modified Kruskal MST approach. Before adding any bridge edge, the angle
between the proposed bridge direction and the last known road direction at
each endpoint is checked. Bridges with angle > 45° are rejected to prevent
geometrically absurd diagonal connections.

Pipeline position:
    skeleton.py → graph_builder.py → healing.py

Input:
    A networkx.Graph where:
      - node attributes: {'o': np.array([row, col])}   ← pixel coords from sknw
      - edge attributes: {'weight': float, 'pts': np.ndarray shape (N,2)}

Output:
    The same graph type with added bridge edges reconnecting components.

Usage:
    from src.graph.healing import heal_graph
    healed_g = heal_graph(g, angle_threshold_deg=45.0, max_fallback=5)
"""

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from scipy.spatial.distance import euclidean

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def heal_graph(
    g: nx.Graph,
    angle_threshold_deg: float = 45.0,
    max_fallback: int = 5,
) -> nx.Graph:
    """
    Heal a disconnected road skeleton graph using MST bridging with
    an angle constraint.

    Parameters
    ----------
    g : nx.Graph
        Graph produced by graph_builder.py / sknw. Node attribute 'o'
        holds the (row, col) pixel coordinate. Edge attribute 'pts' holds
        the ordered array of pixels along the skeleton edge.
    angle_threshold_deg : float
        Maximum allowed angle (degrees) between the last road direction at
        a component endpoint and the proposed bridge direction.
        Default: 45.0
    max_fallback : int
        How many next-nearest candidate pairs to try if the nearest pair
        fails the angle check. Default: 5

    Returns
    -------
    dict with keys:
        "graph"              : nx.Graph  — healed graph with bridge edges added
        "components_before"  : int       — component count before healing
        "components_after"   : int       — component count after healing
        "connectivity_ratio" : float     — (before-after)/before × 100%
                                           100% = fully connected, 0% = no change
    """
    g = g.copy()

    # Record component count BEFORE healing (PS evaluation criterion)
    components_before = nx.number_connected_components(g)

    # Repeat until the graph is fully connected or no valid bridge exists.
    max_iterations = len(g.nodes)  # safety cap
    for _ in range(max_iterations):
        components = list(nx.connected_components(g))
        if len(components) == 1:
            break  # already fully connected

        bridge_added = _connect_nearest_components(
            g, components, angle_threshold_deg, max_fallback
        )
        if not bridge_added:
            remaining = len(list(nx.connected_components(g)))
            print(
                f"[healing] Warning: stopped with {remaining} disconnected "
                f"component(s). No valid bridge found within angle constraint "
                f"and fallback limit={max_fallback}."
            )
            break

    # Record component count AFTER healing
    components_after = nx.number_connected_components(g)

    # connectivity_ratio: % of components eliminated by healing
    # 100% = fully connected, 0% = no improvement
    connectivity_ratio = (
        (components_before - components_after) / components_before * 100
        if components_before > 0 else 0.0
    )

    print(
        f"[healing] Components before: {components_before}  "
        f"after: {components_after}  "
        f"connectivity_ratio: {connectivity_ratio:.1f}%"
    )

    return {
        "graph":               g,
        "components_before":   components_before,
        "components_after":    components_after,
        "connectivity_ratio":  round(connectivity_ratio, 2),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_node_coords(g: nx.Graph) -> dict:
    """
    Return {node_id: np.array([row, col])} for every node.
    Supports pixel_row/pixel_col, sknw-style 'o', and 'y'/'x' attributes.
    """
    coords = {}
    for n, data in g.nodes(data=True):
        if 'pixel_row' in data and 'pixel_col' in data:
            coords[n] = np.array([data['pixel_row'], data['pixel_col']], dtype=float)
        elif 'o' in data:
            coords[n] = np.array(data['o'], dtype=float)
        elif 'y' in data and 'x' in data:
            coords[n] = np.array([data['y'], data['x']], dtype=float)
        else:
            raise KeyError(
                f"Node {n} has no coordinate attribute. "
                "Expected 'pixel_row'/'pixel_col', 'o' (sknw), or 'y'/'x' (geo)."
            )
    return coords


def _last_segment_direction(g: nx.Graph, node: int, coords: dict) -> np.ndarray:
    """
    Compute the direction vector of the last skeleton segment at `node`.

    Strategy:
      1. Look at all edges incident to `node`.
      2. For each edge, read the 'pts' array (ordered pixel path).
         The direction at `node` is from the second-to-last point toward
         the endpoint that IS `node`.
      3. Average the direction vectors (handles degree > 1 nodes).
      4. If no 'pts' available, fall back to the vector toward the
         nearest neighbour node.

    Returns a unit vector (np.ndarray shape (2,)), or None if the node
    is completely isolated (no edges).
    """
    directions = []

    for u, v, data in g.edges(node, data=True):
        pts = data.get('pixel_path', data.get('pts', None))

        if pts is not None and len(pts) >= 2:
            pts = np.array(pts, dtype=float)
            # Determine which end of 'pts' corresponds to `node`.
            # sknw orders pts from u→v; node==u → take first segment,
            # node==v → take last segment (reversed).
            node_coord = coords[node]
            dist_to_start = euclidean(pts[0], node_coord)
            dist_to_end   = euclidean(pts[-1], node_coord)

            if dist_to_start <= dist_to_end:
                # node is at pts[0]; direction = pts[0] → pts[1]
                direction = pts[1] - pts[0]
            else:
                # node is at pts[-1]; direction = pts[-1] → pts[-2]
                direction = pts[-1] - pts[-2]

        else:
            # Fallback: use vector toward the neighbour node
            neighbour = v if u == node else u
            direction = coords[neighbour] - coords[node]

        norm = np.linalg.norm(direction)
        if norm > 0:
            directions.append(direction / norm)

    if not directions:
        return None  # isolated node — no direction available

    mean_dir = np.mean(directions, axis=0)
    norm = np.linalg.norm(mean_dir)
    return mean_dir / norm if norm > 0 else directions[0]


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute the angle in degrees between two 2-D vectors.
    Uses scipy's formula: arccos(dot(v1, v2) / (|v1| * |v2|)).
    Clipped to [0, 180] to guard against floating-point drift.

    This is effectively scipy.spatial.distance.cosine converted to degrees,
    but kept explicit to avoid an extra import and to handle edge cases.
    """
    # Equivalent to scipy's angle_between_vectors in 3 lines:
    dot   = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.degrees(np.arccos(np.clip(dot / norms, -1.0, 1.0)))
    return float(angle)


def _candidate_bridge_pairs(
    comp_a_nodes: list,
    comp_b_nodes: list,
    coords: dict,
    k: int = 10,
) -> list:
    """
    Return up to `k` (node_a, node_b, distance) pairs between two
    components, sorted by Euclidean distance (nearest first).

    Uses a KD-tree on comp_b for fast nearest-neighbour lookup.
    """
    pts_b = np.array([coords[n] for n in comp_b_nodes])
    tree  = cKDTree(pts_b)

    candidates = []
    for na in comp_a_nodes:
        dists, idxs = tree.query(coords[na], k=min(k, len(comp_b_nodes)))
        # tree.query returns a scalar when k=1 — normalise to arrays
        if k == 1:
            dists, idxs = [dists], [idxs]
        for dist, idx in zip(np.atleast_1d(dists), np.atleast_1d(idxs)):
            nb = comp_b_nodes[idx]
            candidates.append((na, nb, float(dist)))

    candidates.sort(key=lambda x: x[2])
    return candidates


def _passes_angle_check(
    g: nx.Graph,
    na: int,
    nb: int,
    coords: dict,
    threshold_deg: float,
) -> bool:
    """
    Check whether the bridge na→nb satisfies the angle constraint at
    both endpoints.

    The bridge direction is the unit vector from coords[na] to coords[nb].

    At na: angle between last road direction and bridge direction ≤ threshold.
    At nb: angle between last road direction and REVERSE bridge direction ≤ threshold.

    If a node has no prior direction (isolated), the check is skipped for
    that endpoint (we allow any bridge to an isolated node).
    """
    bridge_vec = coords[nb] - coords[na]
    norm = np.linalg.norm(bridge_vec)
    if norm == 0:
        return False  # same point — skip
    bridge_unit = bridge_vec / norm

    dir_a = _last_segment_direction(g, na, coords)
    dir_b = _last_segment_direction(g, nb, coords)

    if dir_a is not None:
        angle_a = _angle_between(dir_a, bridge_unit)
        # Allow both "forward" and "backward" alignment
        angle_a = min(angle_a, 180.0 - angle_a)
        if angle_a > threshold_deg:
            return False

    if dir_b is not None:
        angle_b = _angle_between(dir_b, -bridge_unit)
        angle_b = min(angle_b, 180.0 - angle_b)
        if angle_b > threshold_deg:
            return False

    return True


def _connect_nearest_components(
    g: nx.Graph,
    components: list,
    angle_threshold_deg: float,
    max_fallback: int,
) -> bool:
    """
    Find the two closest components by their nearest node pair.
    Try up to `max_fallback` candidate pairs for the angle constraint.
    Add a bridge edge if a valid pair is found.

    Returns True if a bridge was added, False otherwise.
    """
    coords = _get_node_coords(g)

    # Build a flat list of (comp_index, node) for KD-tree search
    best_bridge = None        # (na, nb, dist)
    best_dist   = float('inf')
    best_comp_pair = None     # (i, j) indices into `components`

    # Compare every pair of components; keep track of the globally nearest pair
    # (we only bridge the closest pair per iteration, like Kruskal).
    comp_list = [list(c) for c in components]

    for i in range(len(comp_list)):
        for j in range(i + 1, len(comp_list)):
            pairs = _candidate_bridge_pairs(
                comp_list[i], comp_list[j], coords, k=max_fallback
            )
            if pairs:
                na, nb, dist = pairs[0]  # nearest pair for this comp-pair
                if dist < best_dist:
                    best_dist = dist
                    best_bridge = pairs        # full candidate list
                    best_comp_pair = (i, j)

    if best_bridge is None:
        return False

    # Try candidates in order (nearest first) until angle check passes
    for na, nb, dist in best_bridge[:max_fallback]:
        if _passes_angle_check(g, na, nb, coords, angle_threshold_deg):
            _add_bridge_edge(g, na, nb, dist, coords)
            print(
                f"[healing] Bridge added: node {na} ↔ node {nb}  "
                f"(dist={dist:.2f}px)"
            )
            return True

    # All candidates failed angle check
    print(
        f"[healing] No valid bridge within {angle_threshold_deg}° for "
        f"component pair {best_comp_pair}. "
        f"Tried {min(max_fallback, len(best_bridge))} candidate(s)."
    )
    return False


def _add_bridge_edge(
    g: nx.Graph,
    na: int,
    nb: int,
    dist: float,
    coords: dict,
) -> None:
    """
    Insert a straight bridge edge between na and nb.
    Synthesises a minimal 'pts' array (just the two endpoints) so
    downstream code that expects 'pts' doesn't break.
    """
    synthetic_pts = [list(coords[na]), list(coords[nb])]
    g.add_edge(
        na, nb,
        length_px=dist,
        length_m=dist,
        pixel_path=synthetic_pts,
        bridge=True,
    )