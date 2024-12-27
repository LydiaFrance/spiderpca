"""
spiderpca: Running PCA on spider biomechanics data
"""

from __future__ import annotations

from .data_loading import load_and_process_spider_data
from .data_rotation import undo_body_rotation
from .data_legs import get_all_legs_markers, put_legs_back, make_coxa_origin, unmake_coxa_origin, reflect_legs, combine_legs, restore_leg_positions
from .plot_legs import plot_leg_overlay
from .PCA import run_PCA
from .PCA_figures import (plot_explained, plot_pc_experiment, plot_pc_histogram, 
                         plot_leg_score_hist, plot_leg_score_hist_panelled,
                         plot_leg_pc_timeseries)
from .PCA_scores import get_score_range, create_scores_dataframe
from .PCA_reconstruct import reconstruct

from importlib.metadata import version

__all__ = ("__version__", 
           "load_and_process_spider_data",
           "undo_body_rotation",
           "reflect_legs",
           "combine_legs",
           "restore_leg_positions",
           "get_all_legs_markers",
           "put_legs_back",
           "make_coxa_origin",
           "unmake_coxa_origin",
           "run_PCA",
           "plot_explained",
           "get_score_range",
           "create_scores_dataframe",
           "plot_pc_experiment",
           "reconstruct",
           "plot_pc_histogram",
           "plot_leg_overlay",
            "plot_leg_score_hist",
           "plot_leg_score_hist_panelled",
           "plot_leg_pc_timeseries")
__version__ = version(__name__)
