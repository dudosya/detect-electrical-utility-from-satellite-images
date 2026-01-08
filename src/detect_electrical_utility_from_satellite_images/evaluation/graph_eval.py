"""Visualization utilities for power grid graph inference."""

from pathlib import Path
from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patheffects as patheffects
import matplotlib as mpl
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from numpy.typing import NDArray

from detect_electrical_utility_from_satellite_images.models.graph_inference import (
    PowerGridGraph,
)


def visualize_graph(
    image: NDArray[np.uint8],
    graph: PowerGridGraph,
    segmentation_map: NDArray[np.float32] | None = None,
    gt_centroids: NDArray[np.float32] | None = None,
    gt_adjacency: NDArray[np.int32] | None = None,
    show_scores: bool = True,
    title: str = "Power Grid Graph",
    figsize: tuple[int, int] = (14, 14),
) -> Figure:
    """Visualize the inferred power grid graph overlaid on the image.

    Args:
        image: Background satellite image (H, W, 3).
        graph: Inferred PowerGridGraph.
        segmentation_map: Optional line segmentation probability map.
        gt_centroids: Optional ground truth tower centroids.
        gt_adjacency: Optional ground truth adjacency matrix.
        show_scores: Whether to show edge scores.
        title: Plot title.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2 if segmentation_map is not None else 1, figsize=figsize)
    
    if segmentation_map is not None:
        ax_img, ax_seg = axes
    else:
        ax_img = axes if not hasattr(axes, "__len__") else axes[0]
        ax_seg = None

    # --- Main visualization with graph overlay ---
    ax_img.imshow(image)

    # Draw predicted edges
    if graph.num_nodes > 0:
        edges = graph.edge_list
        for i, j, score in edges:
            x1, y1 = graph.nodes[i]
            x2, y2 = graph.nodes[j]
            
            # Color by score intensity
            color = mpl.colormaps["Reds"](0.3 + 0.7 * score)
            ax_img.plot([x1, x2], [y1, y2], color=color, linewidth=2.5, alpha=0.9, zorder=5)
            
            if show_scores:
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                ax_img.text(
                    mid_x, mid_y, f"{score:.2f}",
                    fontsize=7, color="white", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="red", alpha=0.7),
                    ha="center", va="center",
                )

    # Draw predicted towers
    if graph.num_nodes > 0:
        # Color by detection score
        colors = mpl.colormaps["YlOrRd"](graph.node_scores)
        ax_img.scatter(
            graph.nodes[:, 0], graph.nodes[:, 1],
            c=colors, s=100, marker="o", edgecolors="black",
            linewidths=1.5, label="Predicted Towers", zorder=6,
        )

    # Draw ground truth edges/towers last so they remain visible
    if gt_centroids is not None and gt_adjacency is not None and len(gt_centroids) > 0:
        for i in range(len(gt_centroids)):
            for j in range(i + 1, len(gt_centroids)):
                if gt_adjacency[i, j]:
                    x1, y1 = gt_centroids[i]
                    x2, y2 = gt_centroids[j]
                    (line,) = ax_img.plot(
                        [x1, x2],
                        [y1, y2],
                        color="blue",
                        linewidth=3,
                        alpha=0.95,
                        linestyle="--",
                        zorder=8,
                    )
                    line.set_path_effects(
                        [
                            patheffects.Stroke(linewidth=5, foreground="white", alpha=0.9),
                            patheffects.Normal(),
                        ]
                    )

    if gt_centroids is not None and len(gt_centroids) > 0:
        ax_img.scatter(
            gt_centroids[:, 0],
            gt_centroids[:, 1],
            c="blue",
            s=90,
            marker="^",
            edgecolors="white",
            linewidths=2,
            label="GT Towers",
            zorder=9,
        )

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="none", edgecolor="red", linewidth=2, label="Predicted Edges"),
        mpatches.Patch(facecolor="blue", edgecolor="white", label="GT Towers"),
        Line2D([0], [0], color="blue", linestyle="--", label="GT Edges"),
    ]
    ax_img.legend(handles=legend_elements, loc="upper right")
    ax_img.set_title(f"{title}\nNodes: {graph.num_nodes}, Edges: {graph.num_edges}")
    ax_img.axis("off")

    # --- Segmentation map visualization ---
    if ax_seg is not None and segmentation_map is not None:
        ax_seg.imshow(image, alpha=0.4)
        ax_seg.imshow(segmentation_map, cmap="hot", alpha=0.6, vmin=0, vmax=1)
        
        # Overlay graph on segmentation
        if graph.num_nodes > 0:
            for i, j, _ in graph.edge_list:
                x1, y1 = graph.nodes[i]
                x2, y2 = graph.nodes[j]
                ax_seg.plot([x1, x2], [y1, y2], color="cyan", linewidth=1.5, alpha=0.8)
            
            ax_seg.scatter(
                graph.nodes[:, 0], graph.nodes[:, 1],
                c="cyan", s=60, marker="o", edgecolors="white",
                linewidths=1, zorder=10,
            )

        ax_seg.set_title("Line Segmentation + Graph")
        ax_seg.axis("off")

    plt.tight_layout()
    return fig


def create_inference_summary(
    image: NDArray[np.uint8],
    graph: PowerGridGraph,
    boxes: NDArray[np.float32],
    segmentation_map: NDArray[np.float32],
    figsize: tuple[int, int] = (20, 10),
) -> Figure:
    """Create a comprehensive summary figure of the inference pipeline.

    Args:
        image: Input satellite image.
        graph: Inferred power grid graph.
        boxes: Detected tower bounding boxes.
        segmentation_map: Line segmentation probability map.
        figsize: Figure size.

    Returns:
        Matplotlib figure with 4 subplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].axis("off")

    # 2. Tower detections
    axes[0, 1].imshow(image)
    for box in boxes:
        x1, y1, x2, y2 = box
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            fill=False, edgecolor="red", linewidth=2,
        )
        axes[0, 1].add_patch(rect)
    axes[0, 1].set_title(f"Stage 1: Tower Detection ({len(boxes)} towers)")
    axes[0, 1].axis("off")

    # 3. Line segmentation
    axes[1, 0].imshow(image, alpha=0.3)
    im = axes[1, 0].imshow(segmentation_map, cmap="hot", alpha=0.7, vmin=0, vmax=1)
    axes[1, 0].set_title("Stage 2: Line Segmentation")
    axes[1, 0].axis("off")

    # 4. Final graph
    axes[1, 1].imshow(image)
    if graph.num_nodes > 0:
        # Draw edges
        for i, j, score in graph.edge_list:
            x1, y1 = graph.nodes[i]
            x2, y2 = graph.nodes[j]
            color = mpl.colormaps["Reds"](0.3 + 0.7 * score)
            axes[1, 1].plot([x1, x2], [y1, y2], color=color, linewidth=2)
        
        # Draw nodes
        axes[1, 1].scatter(
            graph.nodes[:, 0], graph.nodes[:, 1],
            c="yellow", s=80, marker="o", edgecolors="red",
            linewidths=1.5, zorder=10,
        )
    
    axes[1, 1].set_title(f"Stage 3: Graph ({graph.num_nodes} nodes, {graph.num_edges} edges)")
    axes[1, 1].axis("off")

    # Shared colorbar so subplot sizes remain symmetric
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)

    plt.suptitle("GridTracer Inference Pipeline", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def save_graph_visualization(
    fig: Figure,
    output_path: Path | str,
    dpi: int = 150,
) -> None:
    """Save graph visualization to file.

    Args:
        fig: Matplotlib figure.
        output_path: Output file path.
        dpi: Resolution for saved image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def visualize_connectivity_scores(
    edge_scores: NDArray[np.float32],
    adjacency: NDArray[np.int32],
    threshold: float = 0.2,
    figsize: tuple[int, int] = (12, 5),
) -> Figure:
    """Visualize edge score distribution and adjacency heatmap.

    Args:
        edge_scores: (N, N) matrix of connectivity scores.
        adjacency: (N, N) binary adjacency matrix.
        threshold: Connectivity threshold used.
        figsize: Figure size.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Extract upper triangle (since symmetric)
    n = edge_scores.shape[0]
    upper_scores = []
    for i in range(n):
        for j in range(i + 1, n):
            if edge_scores[i, j] > 0:  # Only non-zero (within distance)
                upper_scores.append(edge_scores[i, j])

    # Histogram of scores
    if upper_scores:
        axes[0].hist(upper_scores, bins=30, edgecolor="black", alpha=0.7)
        axes[0].axvline(x=threshold, color="red", linestyle="--", linewidth=2, label=f"Threshold={threshold}")
        axes[0].set_xlabel("Connectivity Score")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Edge Score Distribution")
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, "No edges to display", ha="center", va="center")
        axes[0].set_title("Edge Score Distribution")

    # Adjacency heatmap
    if n > 0:
        im = axes[1].imshow(edge_scores, cmap="Reds", vmin=0, vmax=1)
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay adjacency with X markers
        for i in range(n):
            for j in range(n):
                if adjacency[i, j] and i != j:
                    axes[1].plot(j, i, "gx", markersize=8, markeredgewidth=2)
        
        axes[1].set_xlabel("Tower j")
        axes[1].set_ylabel("Tower i")
        axes[1].set_title("Connectivity Scores (X = Connected)")
    else:
        axes[1].text(0.5, 0.5, "No towers detected", ha="center", va="center")
        axes[1].set_title("Connectivity Scores")

    plt.tight_layout()
    return fig
