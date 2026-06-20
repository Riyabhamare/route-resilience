import cv2
import numpy as np
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


def extract_graph_data(graph) -> dict:
    nodes = {
        nid: {"pixel_coord": tuple(data["o"].astype(int))}
        for nid, data in graph.nodes(data=True)
    }
    edges = [
        {
            "from":       u,
            "to":         v,
            "pixel_path": data["pts"].tolist(),
            "length_px":  float(data["weight"]),
        }
        for u, v, data in graph.edges(data=True)
    ]
    return {"nodes": nodes, "edges": edges}


def mask_to_graph(mask: np.ndarray) -> dict:
    cleaned = clean_mask(mask)
    skel    = to_skeleton(cleaned)
    graph   = build_graph(skel)
    return extract_graph_data(graph)