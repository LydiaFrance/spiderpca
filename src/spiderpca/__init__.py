"""
spiderpca: Running PCA on spider biomechanics data
"""

from __future__ import annotations

from .data_loading import load_and_process_spider_data
from .data_rotation import undo_body_rotation
from .PCA import run_PCA
from .PCA_figures import plot_explained, plot_pc_experiment, plot_pc_histogram
from .PCA_scores import get_score_range, create_scores_dataframe
from .PCA_reconstruct import reconstruct

from importlib.metadata import version

__all__ = ("__version__", 
           "load_and_process_spider_data",
           "undo_body_rotation",
           "run_PCA",
           "plot_explained",
           "get_score_range",
           "create_scores_dataframe",
           "plot_pc_experiment",
           "reconstruct",
           "plot_pc_histogram")
__version__ = version(__name__)
