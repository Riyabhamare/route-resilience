import numpy as np
import networkx as nx
from pathlib import Path

try:
    import rasterio
    from rasterio.transform import xy as rasterio_xy
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

from graph.skeleton import mask_to_graph


def pixel_to_latlon(row: int, col: int, transform) -> tuple:
    """Convert pixel (row, col) to (lat, lon) using rasterio transform."""
    lon, lat = rasterio_xy(transform, row, col)
    return float(lat), float(lon)


def build_geo_graph(mask: np.ndarray, transform=None) -> nx.Graph:
    """
    Build a NetworkX graph with geo coordinates on nodes and edges.

    Parameters
    ----------
    mask      : binary mask (np.ndarray, uint8)
    transform : rasterio Affine transform (optional).
                If None, pixel coords are used as-is.

    Returns
    -------
    G : nx.Graph where each node has (lat, lon) attributes
        and each edge has length_px and length_m attributes
    """
    raw = mask_to_graph(mask)

    G = nx.Graph()

    # Add nodes with geo coords
    for node_id, ndata in raw["nodes"].items():
        row, col = ndata["pixel_coord"]

        if transform is not None and HAS_RASTERIO:
            lat, lon = pixel_to_latlon(row, col, transform)
        else:
            # fallback: use pixel coords directly
            lat, lon = float(row), float(col)

        G.add_node(node_id, lat=lat, lon=lon,
                   pixel_row=row, pixel_col=col)

    # Add edges with length
    for edge in raw["edges"]:
        u, v = edge["from"], edge["to"]
        length_px = edge["length_px"]

        # Estimate real-world length if transform available
        if transform is not None and HAS_RASTERIO:
            # pixel size in degrees (approx metres at equator)
            pixel_size_m = abs(transform.a) * 111320
            length_m = length_px * pixel_size_m
        else:
            length_m = length_px  # fallback: same as pixels

        G.add_edge(u, v,
                   length_px=length_px,
                   length_m=length_m,
                   pixel_path=edge["pixel_path"])

    return G


def graph_from_tif(tif_path: str) -> nx.Graph:
    """
    Full pipeline: load a GeoTIFF mask → build geo graph.
    Use this when you have .tif files with geo metadata.
    """
    if not HAS_RASTERIO:
        raise ImportError("pip install rasterio")

    with rasterio.open(tif_path) as src:
        mask = src.read(1)          # first band
        transform = src.transform

    binary_mask = (mask > 0).astype(np.uint8) * 255
    return build_geo_graph(binary_mask, transform)


def graph_from_png(png_path: str) -> nx.Graph:
    """
    Pipeline for plain PNG masks (no geo metadata).
    Nodes get pixel coords as lat/lon fallback.
    """
    from skimage.io import imread
    from skimage.color import rgb2gray

    raw = imread(png_path)
    gray = rgb2gray(raw) if raw.ndim == 3 else raw / 255.0
    mask = (gray > 0.5).astype(np.uint8) * 255

    return build_geo_graph(mask, transform=None)