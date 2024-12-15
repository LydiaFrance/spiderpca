import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_explained(explained_ratio, 
                   ax=None, 
                   colour_before=12, 
                   annotate=True, 
                   xlim_max=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(6.7, 3.5))
        fig.set_constrained_layout(False)

    bar_colors = ['#817', '#a35', '#c66', 
              '#e94', '#ed0', '#9d5', 
              '#4d8', '#2cb', '#0bc',
              '#09c', '#36b', '#639',
              '#817', '#a35', '#c66', 
              '#e94', '#ed0', '#9d5', 
              '#4d8', '#2cb', '#0bc',
              '#09c', '#36b', '#639']


    if colour_before == 0:
        bar_colour = "#51B3D4"
    else:
        bar_colour = "#E6E6E6"

    barlist = plt.bar(range(0,len(explained_ratio)), np.cumsum(explained_ratio), 
        color = bar_colour, alpha = 0.8, width = 0.6, edgecolor='None',zorder = 2)

    for i in range(colour_before):
        barlist[i].set_color(bar_colors[i])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='x', size=0)

    position = ax.get_position()

    if annotate:
        ax_right = ax.twiny()
        ax_right.spines["bottom"].set_position(("axes", 1.05))
        ax_right.spines["bottom"].set_linewidth(1.5)
        ax_right.xaxis.set_ticks_position("bottom")
        ax_right.xaxis.set_tick_params(width=1.5, length=6)
        ax_right.spines["bottom"].set_visible(True)
        ax_right.set_xticks([-1, -0.33, 0.17, 1])
        ax_right.set_xticklabels(['', '', '', ''])
        ax_right.set_xlim(-1, 1)
        
        ax_right.spines['top'].set_visible(False)
        ax_right.spines['right'].set_visible(False)
        ax_right.spines['left'].set_visible(False)

        ax.annotate('>95%', xy=(0.22, 0.94), xycoords='figure fraction')
        ax.annotate('>97%', xy=(0.44, 0.94), xycoords='figure fraction')
        ax.annotate('>98%', xy=(0.68, 0.94), xycoords='figure fraction')

    ax.set_xlabel("Component Number")
    ax.set_ylabel("Cumulative Explained variance ratio")
    
    ax.set_ylim(0,1)
    ax.set_xticks(range(0,len(explained_ratio)))
    ax.set_xticklabels(range(1,len(explained_ratio)+1), fontsize=6)
    ax.set_xlim(-0.5,xlim_max) 

    ax.grid(True, alpha=0.3)
    
    plt.show()

    return ax

def plot_pc_experiment(scores_df, 
                       pc_number=1, 
                       conditions=None, 
                       time_range=(0, 250), 
                       figsize=(8, 3), 
                       alpha=0.5, 
                       marker_size=1):
    """
    Plot PCA scores comparison across different experimental conditions.
    
    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PCA scores and condition information
        Must have columns: ['time_in_frames', f'PC{pc_number}', 'sq_level']
    pc_number : int, optional
        Principal component number to plot (default: 1)
    conditions : list, optional
        List of conditions to compare. If None, uses ["sq040", "sq060", "sq080", "sq100"]
    time_range : tuple, optional
        (min, max) time range to plot (default: (0, 250))
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (8, 3))
    alpha : float, optional
        Transparency of scatter points (default: 0.5)
    marker_size : float, optional
        Size of scatter points (default: 1)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plots
    ax : numpy.ndarray
        Array of subplot axes objects
    """
    if conditions is None:
        conditions = ["sq040", "sq060", "sq080", "sq100"]
    
    fig, ax = plt.subplots(1, len(conditions), figsize=figsize, 
                          sharex=True, sharey=True)
    
    for ii, condition in enumerate(conditions):
        condition_scores = scores_df[scores_df["sq_level"] == condition]
        ax[ii].scatter(condition_scores["time_in_frames"], 
                      condition_scores[f"PC{pc_number}"],
                      marker=".", alpha=alpha, s=marker_size)
        ax[ii].set_title(f"{condition}")
        ax[ii].set_xlabel("time in frames")
        ax[ii].set_xlim(time_range)
    
    # Add y-label to leftmost subplot
    ax[0].set_ylabel(f"PC{pc_number} score")
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    return fig, ax

def plot_pc_histogram(scores_df, pc_number=1, conditions=None, 
                     figsize=(6, 4), alpha=0.5, bins=50, density=True):
    """
    Plot overlaid histograms of PCA scores for different experimental conditions.
    
    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing PCA scores and condition information
        Must have columns: [f'PC{pc_number}', 'sq_level']
    pc_number : int, optional
        Principal component number to plot (default: 1)
    conditions : list, optional
        List of conditions to compare. If None, uses ["sq040", "sq060", "sq080", "sq100"]
    figsize : tuple, optional
        Figure size in inches (width, height) (default: (6, 4))
    alpha : float, optional
        Transparency of histograms (default: 0.5)
    bins : int, optional
        Number of bins for histogram (default: 50)
    density : bool, optional
        If True, normalize histogram to show density instead of counts (default: True)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object containing the plot
    ax : matplotlib.axes.Axes
        The axes object for further customization
    """
    if conditions is None:
        conditions = ["sq040", "sq060", "sq080", "sq100"]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use a color cycle for different conditions
    colors = ["#e94", "#c66", "#a35", "#817"]
    # Plot histogram for each condition
    for condition, color in zip(conditions, colors):
        condition_scores = scores_df[scores_df["sq_level"] == condition]
        ax.hist(condition_scores[f"PC{pc_number}"], 
                bins=bins,
                alpha=alpha,
                color=color,
                label=condition,
                density=density)
    
    ax.set_xlabel(f"PC{pc_number} score")
    ax.set_ylabel("Density" if density else "Count")
    ax.legend(title="Condition")
    
    plt.tight_layout()
    
    return fig, ax