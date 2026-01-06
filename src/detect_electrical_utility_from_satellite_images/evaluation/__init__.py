"""Evaluation modules for GridTracer."""

from detect_electrical_utility_from_satellite_images.evaluation.graph_eval import (
    create_inference_summary,
    save_graph_visualization,
    visualize_connectivity_scores,
    visualize_graph,
)
from detect_electrical_utility_from_satellite_images.evaluation.tower_eval import (
    compute_detection_metrics,
    create_detection_grid,
    evaluate_model_on_dataset,
    save_visualization,
    visualize_detections,
)

__all__ = [
    # Tower evaluation
    "compute_detection_metrics",
    "create_detection_grid",
    "evaluate_model_on_dataset",
    "save_visualization",
    "visualize_detections",
    # Graph evaluation
    "create_inference_summary",
    "save_graph_visualization",
    "visualize_connectivity_scores",
    "visualize_graph",
]
