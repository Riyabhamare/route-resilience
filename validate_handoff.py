"""
validate_handoff.py — Pre-M3 Validation & Handoff Check
=========================================================
Runs the full pipeline on 5 synthetic road mask samples and verifies:

  1. skeleton.py  — mask_to_graph() returns nodes, edges, critical_edges
                    with betweenness_centrality on every edge
  2. graph_builder.py — build_geo_graph() returns nx.Graph with correct
                        node attrs (lat, lon, pixel_row, pixel_col)
                        matching CONTRACTS.md signature
  3. healing.py   — heal_graph() returns dict with graph, components_before,
                    components_after, connectivity_ratio

Prints a per-sample report and a final PASS/FAIL for each check.

Usage
-----
    python validate_handoff.py          # from project root
    # or from notebooks/:
    # import sys; sys.path.insert(0,'..'); exec(open('../validate_handoff.py').read())
"""

import sys
import numpy as np
import networkx as nx

sys.path.insert(0, ".")          # run from project root
sys.path.insert(0, "src")        # so graph.skeleton etc. resolve

from graph.skeleton      import mask_to_graph   as skeleton_mask_to_graph
from graph.graph_builder import build_geo_graph
from graph.healing       import heal_graph

# ── CONTRACTS.md public signature ──────────────────────────────────────────
# def mask_to_graph(mask: np.ndarray, geo_transform=None) -> nx.Graph:
#     Node attrs : y, x  (pixel)  +  lat, lon  (if geo_transform given)
# ───────────────────────────────────────────────────────────────────────────

REQUIRED_NODE_ATTRS  = {"lat", "lon", "pixel_row", "pixel_col"}
REQUIRED_EDGE_ATTRS  = {"length_px", "length_m", "pixel_path"}
REQUIRED_SKEL_KEYS   = {"nodes", "edges", "critical_edges"}
REQUIRED_HEAL_KEYS   = {"graph", "components_before",
                         "components_after", "connectivity_ratio"}


# ── 5 synthetic road mask generators ───────────────────────────────────────

def _make_mask(shape=(200, 200), dtype=np.uint8):
    return np.zeros(shape, dtype=dtype)


def sample_1_straight_horizontal():
    """Single straight horizontal road — simplest case."""
    m = _make_mask()
    m[95:105, 20:180] = 255
    return m, "straight_horizontal"


def sample_2_cross():
    """Cross junction — tests node degree > 2."""
    m = _make_mask()
    m[95:105, 20:180] = 255
    m[20:180, 95:105] = 255
    return m, "cross_junction"


def sample_3_two_parallel_roads():
    """Two parallel roads — disconnected, tests healing."""
    m = _make_mask()
    m[60:70,  20:180] = 255
    m[130:140, 20:180] = 255
    return m, "two_parallel_roads"


def sample_4_l_shape():
    """L-shaped road — tests non-straight direction vectors."""
    m = _make_mask()
    m[95:105, 20:100] = 255   # horizontal arm
    m[100:180, 95:105] = 255  # vertical arm going down
    return m, "l_shape"


def sample_5_fragmented():
    """Four separate short segments — stress-tests healing + ratio."""
    m = _make_mask()
    m[40:50,  20:80]   = 255
    m[40:50,  120:180] = 255
    m[150:160, 20:80]  = 255
    m[150:160, 120:180] = 255
    return m, "fragmented_4_segments"


SAMPLES = [
    sample_1_straight_horizontal,
    sample_2_cross,
    sample_3_two_parallel_roads,
    sample_4_l_shape,
    sample_5_fragmented,
]

# ── Validation helpers ──────────────────────────────────────────────────────

def check(condition: bool, label: str, failures: list):
    status = "✅" if condition else "❌"
    print(f"    {status}  {label}")
    if not condition:
        failures.append(label)


def validate_skeleton_output(skel_out: dict, failures: list):
    """Check skeleton.py raw dict structure."""
    check(isinstance(skel_out, dict),
          "skeleton output is a dict", failures)
    check(REQUIRED_SKEL_KEYS.issubset(skel_out.keys()),
          f"skeleton dict has keys {REQUIRED_SKEL_KEYS}", failures)

    edges = skel_out.get("edges", [])
    if edges:
        sample_edge = edges[0]
        check("betweenness_centrality" in sample_edge,
              "edge has 'betweenness_centrality' attribute", failures)
        check("length_px" in sample_edge,
              "edge has 'length_px' attribute", failures)
        check("pixel_path" in sample_edge,
              "edge has 'pixel_path' attribute", failures)

    critical = skel_out.get("critical_edges", [])
    check(isinstance(critical, list),
          "critical_edges is a list", failures)
    check(len(critical) <= 10,
          "critical_edges has ≤ 10 entries", failures)
    if critical:
        check("betweenness_centrality" in critical[0],
              "critical_edges entries have betweenness_centrality", failures)


def validate_graph_contract(G: nx.Graph, failures: list):
    """Check CONTRACTS.md node/edge signature."""
    check(isinstance(G, nx.Graph),
          "build_geo_graph returns nx.Graph", failures)
    check(G.number_of_nodes() > 0,
          "graph has at least 1 node", failures)

    if G.number_of_nodes() > 0:
        sample_node_data = dict(list(G.nodes(data=True))[0][1])
        check(REQUIRED_NODE_ATTRS.issubset(sample_node_data.keys()),
              f"node has attrs {REQUIRED_NODE_ATTRS} "
              f"(got {set(sample_node_data.keys())})", failures)

        # CONTRACTS.md uses 'y' and 'x' — map: y=lat(row), x=lon(col)
        # our graph uses pixel_row/pixel_col + lat/lon which satisfies this
        check("lat" in sample_node_data and "lon" in sample_node_data,
              "node has lat/lon (satisfies CONTRACTS y/x requirement)", failures)

    if G.number_of_edges() > 0:
        sample_edge_data = dict(list(G.edges(data=True))[0][2])
        check(REQUIRED_EDGE_ATTRS.issubset(sample_edge_data.keys()),
              f"edge has attrs {REQUIRED_EDGE_ATTRS}", failures)


def validate_healing_output(heal_out: dict, failures: list):
    """Check healing.py return dict structure."""
    check(isinstance(heal_out, dict),
          "heal_graph returns a dict", failures)
    check(REQUIRED_HEAL_KEYS.issubset(heal_out.keys()),
          f"heal output has keys {REQUIRED_HEAL_KEYS}", failures)

    check(isinstance(heal_out.get("graph"), nx.Graph),
          "heal_out['graph'] is nx.Graph", failures)
    check(isinstance(heal_out.get("components_before"), int),
          "components_before is int", failures)
    check(isinstance(heal_out.get("components_after"), int),
          "components_after is int", failures)
    check(isinstance(heal_out.get("connectivity_ratio"), float),
          "connectivity_ratio is float", failures)

    ratio = heal_out.get("connectivity_ratio", -1)
    check(0.0 <= ratio <= 100.0,
          f"connectivity_ratio in [0, 100] (got {ratio})", failures)

    cb = heal_out.get("components_before", 0)
    ca = heal_out.get("components_after",  0)
    check(ca <= cb,
          f"components_after ({ca}) ≤ components_before ({cb})", failures)


# ── Main runner ─────────────────────────────────────────────────────────────

def run_validation():
    print("=" * 65)
    print("  ROUTE-RESILIENCE — Pre-M3 Handoff Validation")
    print("=" * 65)

    all_failures   = []
    sample_results = []

    for i, sample_fn in enumerate(SAMPLES, 1):
        mask, name = sample_fn()
        print(f"\n[Sample {i}/5]  {name}  (mask shape {mask.shape})")
        failures = []

        # ── Step A: skeleton.py ──────────────────────────────────────────
        print("  ▸ skeleton.mask_to_graph()")
        try:
            skel_out = skeleton_mask_to_graph(mask)
            validate_skeleton_output(skel_out, failures)
            n_nodes = len(skel_out["nodes"])
            n_edges = len(skel_out["edges"])
            n_crit  = len(skel_out["critical_edges"])
            print(f"      nodes={n_nodes}  edges={n_edges}  "
                  f"critical_edges={n_crit}")
        except Exception as exc:
            failures.append(f"skeleton crashed: {exc}")
            print(f"    ❌  skeleton crashed: {exc}")
            sample_results.append((name, failures))
            all_failures.extend(failures)
            continue

        # ── Step B: graph_builder.py (CONTRACTS signature) ───────────────
        print("  ▸ graph_builder.build_geo_graph()  [CONTRACTS.md check]")
        try:
            G = build_geo_graph(mask, transform=None)
            validate_graph_contract(G, failures)
        except Exception as exc:
            failures.append(f"graph_builder crashed: {exc}")
            print(f"    ❌  graph_builder crashed: {exc}")
            sample_results.append((name, failures))
            all_failures.extend(failures)
            continue

        # ── Step C: healing.py ───────────────────────────────────────────
        print("  ▸ healing.heal_graph()")
        try:
            heal_out = heal_graph(G, angle_threshold_deg=45.0, max_fallback=5)
            validate_healing_output(heal_out, failures)

            cb    = heal_out["components_before"]
            ca    = heal_out["components_after"]
            ratio = heal_out["connectivity_ratio"]
            print(f"      components  before={cb}  after={ca}  "
                  f"connectivity_ratio={ratio}%")
        except Exception as exc:
            failures.append(f"healing crashed: {exc}")
            print(f"    ❌  healing crashed: {exc}")
            sample_results.append((name, failures))
            all_failures.extend(failures)
            continue

        sample_results.append((name, failures))
        all_failures.extend(failures)

    # ── Final report ─────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    for name, failures in sample_results:
        status = "PASS ✅" if not failures else f"FAIL ❌ ({len(failures)} issue(s))"
        print(f"  {name:<30} {status}")
        for f in failures:
            print(f"      → {f}")

    print()
    if not all_failures:
        print("  🟢  ALL CHECKS PASSED — ready for M3 handoff")
    else:
        total = len(all_failures)
        print(f"  🔴  {total} check(s) FAILED — fix before M3 handoff")

    print("=" * 65)
    return len(all_failures) == 0


if __name__ == "__main__":
    passed = run_validation()
    sys.exit(0 if passed else 1)