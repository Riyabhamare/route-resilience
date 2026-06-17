# CONTRACTS.md

## Member 1 provides: src/models/inference.py

```python
def predict_mask(image_path: str) -> np.ndarray:
    """Returns binary mask, same H x W as input, dtype uint8 (0/255)."""
```

## Member 2 provides: src/graph/graph_builder.py

```python
def mask_to_graph(mask: np.ndarray, geo_transform=None) -> nx.Graph:
    """Returns healed, connected graph.
    Node attrs: y, x (pixel), lat, lon (if geo_transform given)."""
```

## Member 3 provides: src/graph/analysis.py

```python
def analyze_graph(G: nx.Graph) -> dict:
    """Returns {"top_bottlenecks": list[dict], "rsi": float,
                 "flood_layer": np.ndarray | None}"""
```

## Member 4 workflow

```python
mask = predict_mask(uploaded_path)
graph = mask_to_graph(mask, geo_transform)
result = analyze_graph(graph)
```
