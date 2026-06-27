import networkx as nx
import json
import os

def analyze_graph(G):
    """
    Takes a healed road graph from Member 2.
    Returns resilience intelligence.
    """

    # --- Step 1: Find critical NODES (intersections) ---
    node_centrality = nx.betweenness_centrality(G, weight='length_m')

    top_nodes = sorted(node_centrality, key=node_centrality.get, reverse=True)[:10]

    top_bottlenecks = []
    for node in top_nodes:
        top_bottlenecks.append({
            "node_id":     node,
            "lat":         G.nodes[node].get('lat'),
            "lon":         G.nodes[node].get('lon'),
            "betweenness": node_centrality[node]
        })

    # --- Step 2: Find critical EDGES (road segments) ---
    critical_edges = []
    for u, v, data in G.edges(data=True):
        critical_edges.append({
            "from": u,
            "to":   v,
            "betweenness": data.get('betweenness_centrality', 0),
            "length_m":    data.get('length_m', 0)
        })
    critical_edges = sorted(critical_edges, key=lambda x: x['betweenness'], reverse=True)[:10]

    # --- Step 3: Compute Network Resilience Ratio (NRR) ---
    # Credit: Latora & Marchiori (2001)
    original_efficiency = nx.global_efficiency(G)

    # --- Step 4: Ablation — what happens when each top node is removed? ---
    ablation_results = []
    for node in top_nodes:
        G_temp = G.copy()
        G_temp.remove_node(node)
        post_efficiency = nx.global_efficiency(G_temp)
        if post_efficiency > 0:
            nrr = original_efficiency / post_efficiency
        else:
            nrr = float('inf')
        ablation_results.append({
            "node_id": node,
            "nrr":     round(nrr, 4)
        })

    # --- Step 5: Package everything ---
    result = {
        "top_bottlenecks":  top_bottlenecks,
        "critical_edges":   critical_edges,
        "nrr":              round(original_efficiency, 4),
        "ablation_results": ablation_results,
    }

    # --- Step 6: Save to outputs folder ---
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/resilience_report.json", "w") as f:
        json.dump(result, f, indent=2)

    print("✅ analyze_graph() done! Results saved to outputs/resilience_report.json")
    return result
def compute_inverse_ablation(G):
    """
    Find the best roads to BUILD to improve network resilience.
    Tries adding edges between disconnected or far-apart nodes
    and measures NRR improvement.
    """
    import itertools

    original_efficiency = nx.global_efficiency(G)
    nodes = list(G.nodes())
    scenarios = []

    # Get all pairs of nodes that are NOT already connected by a direct edge
    non_edges = [(u, v) for u, v in itertools.combinations(nodes, 2)
                 if not G.has_edge(u, v)]

    # Try adding each potential road and measure improvement
    for u, v in non_edges:
        # Estimate distance between the two nodes
        lat1, lon1 = G.nodes[u]['lat'], G.nodes[u]['lon']
        lat2, lon2 = G.nodes[v]['lat'], G.nodes[v]['lon']
        distance = ((lat2 - lat1)**2 + (lon2 - lon1)**2) ** 0.5

        # Add the road temporarily
        G_temp = G.copy()
        G_temp.add_edge(u, v, length_m=distance, length_px=distance)

        # Measure new efficiency
        new_efficiency = nx.global_efficiency(G_temp)

        # Calculate improvement percentage
        if original_efficiency > 0:
            improvement = (new_efficiency - original_efficiency) / original_efficiency * 100
        else:
            improvement = 0

        scenarios.append({
            "from_node":      u,
            "to_node":        v,
            "from_lat":       lat1,
            "from_lon":       lon1,
            "to_lat":         lat2,
            "to_lon":         lon2,
            "nrr_improvement_pct": round(improvement, 2)
        })

    # Sort by most improvement and keep top 5
    scenarios = sorted(scenarios, key=lambda x: x['nrr_improvement_pct'], reverse=True)[:5]

    # Save to file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/inverse_ablation_results.json", "w") as f:
        json.dump(scenarios, f, indent=2)

    print("✅ inverse ablation done! Saved to outputs/inverse_ablation_results.json")
    return scenarios
def compute_flood_risk(G, dem_path=None, ndwi_path=None):
    """
    Computes flood risk score for each road node.
    
    If real elevation (DEM) and water (NDWI) data files are provided, uses them.
    Otherwise falls back to a simulation using node position.
    
    Score formula: betweenness x ndwi_risk x (1 / (elevation + 1))
    Low elevation + near water + high traffic = highest risk.
    """

    # Get betweenness centrality for all nodes
    node_centrality = nx.betweenness_centrality(G, weight='length_m')

    flood_scores = {}

    for node, data in G.nodes(data=True):
        lat = data.get('lat', 0)
        lon = data.get('lon', 0)
        betweenness = node_centrality.get(node, 0)

        if dem_path and ndwi_path:
            # Real data mode — sample elevation and water risk from raster files
            try:
                import rasterio
                with rasterio.open(dem_path) as dem:
                    elevation = list(dem.sample([(lon, lat)]))[0][0]
                with rasterio.open(ndwi_path) as ndwi:
                    ndwi_risk = list(ndwi.sample([(lon, lat)]))[0][0]
            except Exception:
                elevation = 10.0
                ndwi_risk = 0.5
        else:
            # Fallback simulation mode
            # Nodes near the centre of the map = lower elevation (valley)
            # Nodes at the edges = higher elevation
            elevation = abs(lat - 100) + abs(lon - 100)
            ndwi_risk = 1.0 / (1.0 + elevation / 50)

        # Final flood vulnerability score
        score = betweenness * ndwi_risk * (1.0 / (elevation + 1))
        flood_scores[node] = {
            "node_id":    node,
            "lat":        lat,
            "lon":        lon,
            "elevation":  round(float(elevation), 2),
            "ndwi_risk":  round(float(ndwi_risk), 4),
            "betweenness": round(betweenness, 4),
            "flood_score": round(score, 6)
        }

    # Sort by highest flood risk
    ranked = sorted(flood_scores.values(), key=lambda x: x['flood_score'], reverse=True)

    # Save to file
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/flood_risk.json", "w") as f:
        json.dump(ranked, f, indent=2)

    print("✅ flood risk done! Saved to outputs/flood_risk.json")
    return ranked