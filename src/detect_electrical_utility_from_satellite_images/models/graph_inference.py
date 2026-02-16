"""Stage 3: Graph Inference - Connect detected towers using line segmentation.

This module implements the graph inference algorithm from the GridTracer paper.
It combines tower detections (Stage 1) with line segmentation probability maps
(Stage 2) to construct a geospatial graph representing the power grid.

Algorithm:
    For each pair of detected towers (i, j):
    1. Check if distance(i, j) < max_distance_m (default: 600m)
    2. Compute connectivity score S_ij by averaging line probabilities along path
    3. Create edge if S_ij >= connectivity_threshold (default: 0.2)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from detect_electrical_utility_from_satellite_images.logging_config import get_logger


@dataclass
class PowerGridGraph:
    """Represents a detected power grid as a graph structure.

    Attributes:
        nodes: (N, 2) array of tower centroid coordinates [x, y] in pixels.
        adjacency: (N, N) binary adjacency matrix.
        node_scores: (N,) confidence scores for each tower detection.
        edge_scores: (N, N) connectivity scores for each potential edge.
        resolution_m_per_px: Image resolution in meters per pixel.
    """

    nodes: NDArray[np.float32]
    adjacency: NDArray[np.int32]
    node_scores: NDArray[np.float32]
    edge_scores: NDArray[np.float32]
    resolution_m_per_px: float = 0.3

    @property
    def num_nodes(self) -> int:
        """Number of towers in the graph."""
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        """Number of power line connections."""
        return int(np.sum(self.adjacency) // 2)  # Undirected, so divide by 2

    @property
    def edge_list(self) -> list[tuple[int, int, float]]:
        """Get list of edges as (node_i, node_j, score) tuples."""
        edges = []
        n = len(self.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if self.adjacency[i, j]:
                    edges.append((i, j, float(self.edge_scores[i, j])))
        return edges

    def to_dict(self) -> dict[str, Any]:
        """Convert graph to dictionary for serialization."""
        return {
            "nodes": self.nodes.tolist(),
            "adjacency": self.adjacency.tolist(),
            "node_scores": self.node_scores.tolist(),
            "edge_scores": self.edge_scores.tolist(),
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "resolution_m_per_px": self.resolution_m_per_px,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PowerGridGraph":
        """Create graph from dictionary."""
        return cls(
            nodes=np.array(data["nodes"], dtype=np.float32),
            adjacency=np.array(data["adjacency"], dtype=np.int32),
            node_scores=np.array(data["node_scores"], dtype=np.float32),
            edge_scores=np.array(data["edge_scores"], dtype=np.float32),
            resolution_m_per_px=data.get("resolution_m_per_px", 0.3),
        )


def compute_path_pixels(
    p1: tuple[float, float],
    p2: tuple[float, float],
    width: int = 9,
) -> NDArray[np.int32]:
    """Get pixel coordinates along a path between two points.

    Uses Bresenham-like approach to get all pixels within a line of given width.

    Args:
        p1: Start point (x, y).
        p2: End point (x, y).
        width: Line width in pixels.

    Returns:
        (M, 2) array of [x, y] pixel coordinates along the path.
    """
    x1, y1 = p1
    x2, y2 = p2

    # Compute line direction and length
    dx = x2 - x1
    dy = y2 - y1
    length = np.sqrt(dx**2 + dy**2)

    if length < 1:
        return np.array([[int(x1), int(y1)]], dtype=np.int32)

    # Number of sample points along line (supersample to reduce aliasing)
    n_samples = max(int(length * 2) + 1, 2)

    # Generate points along the center line
    t = np.linspace(0, 1, n_samples)
    center_x = x1 + t * dx
    center_y = y1 + t * dy

    # Compute perpendicular direction for width
    if length > 0:
        perp_x = -dy / length
        perp_y = dx / length
    else:
        perp_x, perp_y = 0, 0

    # Generate points across the width
    half_width = width / 2
    offsets = np.linspace(-half_width, half_width, max(1, width))

    pixels = []
    for ox in offsets:
        px = center_x + ox * perp_x
        py = center_y + ox * perp_y
        for x, y in zip(px, py, strict=False):
            pixels.append([int(round(x)), int(round(y))])

    # Remove duplicates
    pixels = np.array(pixels, dtype=np.int32)
    if len(pixels) > 0:
        pixels = np.unique(pixels, axis=0)

    return pixels


def compute_connectivity_score(
    segmentation_map: NDArray[np.float32],
    tower_i: tuple[float, float],
    tower_j: tuple[float, float],
    line_width: int = 9,
) -> float:
    """Compute connectivity score between two towers.

    The score is the average probability value along the path between towers.

    Args:
        segmentation_map: (H, W) probability map from line segmentation.
        tower_i: First tower centroid (x, y) in pixels.
        tower_j: Second tower centroid (x, y) in pixels.
        line_width: Width of the path to sample (default: 9px from paper).

    Returns:
        Average probability along the path (0.0 to 1.0).
    """
    h, w = segmentation_map.shape

    # Get pixels along the path
    path_pixels = compute_path_pixels(tower_i, tower_j, width=line_width)

    if len(path_pixels) == 0:
        return 0.0

    # Filter out-of-bounds pixels
    valid_mask = (
        (path_pixels[:, 0] >= 0)
        & (path_pixels[:, 0] < w)
        & (path_pixels[:, 1] >= 0)
        & (path_pixels[:, 1] < h)
    )
    valid_pixels = path_pixels[valid_mask]

    if len(valid_pixels) == 0:
        return 0.0

    # Sample segmentation values
    values = segmentation_map[valid_pixels[:, 1], valid_pixels[:, 0]]

    return float(np.mean(values))


def compute_distance_matrix(
    centroids: NDArray[np.float32],
    resolution_m_per_px: float = 0.3,
) -> NDArray[np.float32]:
    """Compute pairwise distance matrix between tower centroids.

    Args:
        centroids: (N, 2) array of [x, y] coordinates in pixels.
        resolution_m_per_px: Image resolution (meters per pixel).

    Returns:
        (N, N) distance matrix in meters.
    """
    n = len(centroids)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # Compute pairwise Euclidean distances in pixels
    diff = centroids[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    dist_px = np.sqrt(np.sum(diff**2, axis=2))

    # Convert to meters
    return dist_px * resolution_m_per_px


def infer_graph(
    tower_centroids: NDArray[np.float32] | torch.Tensor,
    tower_scores: NDArray[np.float32] | torch.Tensor,
    segmentation_map: NDArray[np.float32] | torch.Tensor,
    max_distance_m: float = 600.0,
    connectivity_threshold: float = 0.2,
    line_width: int = 9,
    resolution_m_per_px: float = 0.3,
) -> PowerGridGraph:
    """Infer power grid graph from tower detections and line segmentation.

    This is the main Stage 3 algorithm from the GridTracer paper:
    1. For each pair of towers, check distance constraint
    2. Compute connectivity score from segmentation map
    3. Create edges where both constraints are satisfied

    Args:
        tower_centroids: (N, 2) array of [x, y] tower coordinates in pixels.
        tower_scores: (N,) array of detection confidence scores.
        segmentation_map: (H, W) line probability map (0-1).
        max_distance_m: Maximum distance for connections (default: 600m).
        connectivity_threshold: Minimum score for edge creation (default: 0.2).
        line_width: Path width for score computation (default: 9px).
        resolution_m_per_px: Image resolution (default: 0.3 m/px).

    Returns:
        PowerGridGraph with detected towers and inferred connections.
    """
    log = get_logger()

    # Convert to numpy if needed
    if isinstance(tower_centroids, torch.Tensor):
        tower_centroids = tower_centroids.cpu().numpy()
    if isinstance(tower_scores, torch.Tensor):
        tower_scores = tower_scores.cpu().numpy()
    if isinstance(segmentation_map, torch.Tensor):
        segmentation_map = segmentation_map.cpu().numpy()

    # Handle 3D/4D segmentation map (squeeze batch/channel dims)
    if segmentation_map.ndim in (3, 4):
        segmentation_map = segmentation_map.squeeze()
    if segmentation_map.ndim != 2:
        raise ValueError(
            "segmentation_map must be 2D after squeezing, got shape "
            f"{segmentation_map.shape}"
        )

    n = len(tower_centroids)
    log.info(
        "graph_inference_start",
        num_towers=n,
        max_distance_m=max_distance_m,
        connectivity_threshold=connectivity_threshold,
    )

    if n == 0:
        return PowerGridGraph(
            nodes=np.zeros((0, 2), dtype=np.float32),
            adjacency=np.zeros((0, 0), dtype=np.int32),
            node_scores=np.zeros((0,), dtype=np.float32),
            edge_scores=np.zeros((0, 0), dtype=np.float32),
            resolution_m_per_px=resolution_m_per_px,
        )

    # Initialize adjacency and edge score matrices
    adjacency = np.zeros((n, n), dtype=np.int32)
    edge_scores = np.zeros((n, n), dtype=np.float32)

    # Process each pair of towers
    n_candidates = 0
    n_connected = 0

    max_distance_px = max_distance_m / resolution_m_per_px
    bin_size = max(1.0, max_distance_px)
    bin_map: dict[tuple[int, int], list[int]] = {}

    for idx, (x, y) in enumerate(tower_centroids):
        bin_x = int(x // bin_size)
        bin_y = int(y // bin_size)
        bin_map.setdefault((bin_x, bin_y), []).append(idx)

    for i in range(n):
        x, y = tower_centroids[i]
        bin_x = int(x // bin_size)
        bin_y = int(y // bin_size)

        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates = bin_map.get((bin_x + dx, bin_y + dy))
                if not candidates:
                    continue
                for j in candidates:
                    if j <= i:
                        continue

                    # Check distance constraint in pixels
                    dist_px = float(np.linalg.norm(tower_centroids[i] - tower_centroids[j]))
                    if dist_px > max_distance_px:
                        continue

                    n_candidates += 1

                    # Compute connectivity score
                    score = compute_connectivity_score(
                        segmentation_map,
                        tuple(tower_centroids[i]),
                        tuple(tower_centroids[j]),
                        line_width=line_width,
                    )

                    # Store score (symmetric)
                    edge_scores[i, j] = score
                    edge_scores[j, i] = score

                    # Check connectivity threshold
                    if score >= connectivity_threshold:
                        adjacency[i, j] = 1
                        adjacency[j, i] = 1
                        n_connected += 1

    log.info(
        "graph_inference_complete",
        num_towers=n,
        candidate_pairs=n_candidates,
        connected_pairs=n_connected,
    )

    return PowerGridGraph(
        nodes=tower_centroids.astype(np.float32),
        adjacency=adjacency,
        node_scores=tower_scores.astype(np.float32),
        edge_scores=edge_scores,
        resolution_m_per_px=resolution_m_per_px,
    )


def run_full_inference(
    image: NDArray[np.uint8] | torch.Tensor,
    tower_model: torch.nn.Module,
    line_model: torch.nn.Module,
    config: Any,
    device: torch.device | None = None,
) -> tuple[PowerGridGraph, dict[str, Any]]:
    """Run the complete 3-stage GridTracer pipeline on an image.

    Args:
        image: Input satellite image (H, W, 3) or (3, H, W).
        tower_model: Stage 1 tower detection model (Faster R-CNN).
        line_model: Stage 2 line segmentation model (U-Net).
        config: Configuration object with inference parameters.
        device: Compute device.

    Returns:
        Tuple of (PowerGridGraph, intermediate_results dict).
    """
    log = get_logger()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Convert image to tensor if needed
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 3:
            # (H, W, 3) -> (3, H, W)
            image = np.transpose(image, (2, 0, 1))
        image = torch.from_numpy(image).float() / 255.0

    # Ensure image is on device
    if image.dim() == 3:
        image = image.unsqueeze(0)  # Add batch dim
    image = image.to(device)

    # --- Stage 1: Tower Detection ---
    log.info("stage_1_tower_detection")
    tower_model.eval()
    with torch.no_grad():
        tower_outputs = tower_model(image)

    # Extract detections (handle both single image and batch)
    if isinstance(tower_outputs, list):
        tower_output = tower_outputs[0]
    else:
        tower_output = tower_outputs

    boxes = tower_output["boxes"]
    scores = tower_output["scores"]
    
    # Filter by confidence threshold
    conf_thresh = config.tower_detection.confidence_threshold
    mask = scores >= conf_thresh
    boxes = boxes[mask]
    scores = scores[mask]

    # Compute centroids
    centroids = torch.stack([
        (boxes[:, 0] + boxes[:, 2]) / 2,
        (boxes[:, 1] + boxes[:, 3]) / 2,
    ], dim=1) if len(boxes) > 0 else torch.zeros((0, 2), device=device)

    log.info("stage_1_complete", num_towers=len(centroids))

    # --- Stage 2: Line Segmentation ---
    log.info("stage_2_line_segmentation")
    line_model.eval()
    with torch.no_grad():
        seg_output = line_model(image)

    # Sigmoid if needed (model might output logits)
    if seg_output.min() < 0 or seg_output.max() > 1:
        seg_output = torch.sigmoid(seg_output)

    # Remove batch/channel dims -> (H, W)
    seg_map = seg_output.squeeze()

    log.info("stage_2_complete", seg_shape=list(seg_map.shape))

    # --- Stage 3: Graph Inference ---
    log.info("stage_3_graph_inference")
    graph = infer_graph(
        tower_centroids=centroids,
        tower_scores=scores,
        segmentation_map=seg_map,
        max_distance_m=config.graph_inference.max_distance_m,
        connectivity_threshold=config.graph_inference.connectivity_threshold,
        line_width=config.line_segmentation.line_width_inference,
        resolution_m_per_px=config.graph_inference.resolution_m_per_px,
    )

    # Collect intermediate results for visualization
    intermediate = {
        "boxes": boxes.cpu().numpy(),
        "scores": scores.cpu().numpy(),
        "centroids": centroids.cpu().numpy(),
        "segmentation_map": seg_map.cpu().numpy(),
    }

    return graph, intermediate
