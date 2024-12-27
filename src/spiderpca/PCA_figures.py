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
    
    # bar_colors = ['#B5E675', '#6ED8A9', '#51B3D4', 
    #           '#4579AA', '#F19EBA', '#BC96C9', 
    #           '#917AC2', '#BE607F', '#624E8B',
    #           '#E6E6E6', '#E6E6E6', '#E6E6E6']


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
                     figsize=(6, 4), alpha=0.5, bins=50, density=True,
                     ax = None):
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
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        toggle_legend = True
    else:
        toggle_legend = False
    
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
    
    if toggle_legend:
        ax.legend(title="Condition", frameon=False)
    
    plt.tight_layout()
    
    return ax



def plot_leg_score_hist(reshaped_scores, pc_number=0, figsize=(6, 4), leg_list=None):
    """
    Plot histogram distributions of PCA scores for each leg.
    
    Parameters
    ----------
    reshaped_scores : ndarray
        Scores array of shape (n_frames, n_legs, n_components)
    pc_number : int, optional
        Principal component to plot (default: 0)
    figsize : tuple, optional
        Figure size in inches (default: (6, 4))
        
    Returns
    -------
    tuple
        matplotlib figure and axis objects
    """
    
    # Color scheme for different legs
    colourList = ["#FC9CF9", "#73E6E2", "#ADD487", "#5F62BF",
                  "#C5C6E8", "#d6e9c3", "#BAF3F1", "#fdc4fb"]
    
    if leg_list is None:
        leg_list = range(reshaped_scores.shape[1])
    
    fig, ax = plt.subplots(figsize=figsize)
    

    # Plot histogram for each leg
    for leg_idx in leg_list:
        ax.hist(reshaped_scores[:, leg_idx, pc_number],
                bins=50,
                alpha=0.6,
                color=colourList[leg_idx],
                label=f"Leg {leg_idx+1}",
                density=True)
    
    ax.set_title(f"PC{pc_number+1} score")
    ax.set_xlabel(f"PC{pc_number+1} score")
    ax.set_ylabel("Density")
    ax.legend(title="Leg")
    
    plt.tight_layout()
    
    return fig, ax


def plot_leg_score_hist_panelled(reshaped_scores, pc_number=0, figsize=(6, 6), pairings=None):
    """
    Plot histogram distributions of PCA scores for leg pairings in 2x2 subplots.
    
    Parameters
    ----------
    reshaped_scores : ndarray
        Scores array of shape (n_frames, n_legs, n_components)
    pc_number : int, optional
        Principal component to plot (default: 0)
    figsize : tuple, optional
        Figure size in inches (default: (12, 8))
    pairings : list of lists, optional
        List of pairings to highlight in each subplot (default: [[0, 7], [1, 6], [2, 5], [3, 4]])
        
    Returns
    -------
    tuple
        matplotlib figure and axis objects
    """
    # Default pairings
    if pairings is None:
        pairings = [[0, 7],  [3, 4], [1, 6], [2, 5]]

        pair_names = ["Front legs","Back legs",
                      "Mid Front legs", "Mid Back legs"]

    else:
        pair_names = [f"Leg {pairing[0]+1} & Leg {pairing[1]+1}" for pairing in pairings]


    # Color scheme for highlighted legs
    colourList = ["#FC9CF9", "#73E6E2", "#ADD487", "#5F62BF",
                  "#C5C6E8", "#d6e9c3", "#BAF3F1", "#fdc4fb"]
    gray_color = "#D3D3D3"  # Color for non-highlighted legs
    
    # Create a 2x2 grid of subplots
    fig, ax = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    ax = ax.flatten()

    
    # Loop through each pairing and create the corresponding subplot
    for i, pairing in enumerate(pairings):
        # Plot gray histograms for all non-highlighted legs first
        for leg_idx in range(reshaped_scores.shape[1]):
            if leg_idx not in pairing:
                ax[i].hist(
                    reshaped_scores[:, leg_idx, pc_number],
                    bins=50,
                    alpha=0.4,
                    color=gray_color,
                    density=True,
                )
        
        # Plot colorful histograms for the highlighted legs on top
        for leg_idx in pairing:
            ax[i].hist(
                reshaped_scores[:, leg_idx, pc_number],
                bins=50,
                alpha=1,
                color=colourList[leg_idx],
                density=True,
                label=f"Leg {leg_idx+1}"
            )
        
        # Set labels and legend
        ax[i].set_xlabel(f"PC{pc_number+1} Score")
        ax[i].set_ylabel("Density")
        ax[i].legend(loc="upper left", fontsize="small", frameon=False)
        ax[i].set_title(pair_names[i])
    
    plt.tight_layout()
    return fig, ax


def plot_leg_pc_timeseries(scores_df, 
                           sequence_id, 
                           components_list=None, 
                           leg_list=None,
                           figsize=(6, 4)):
    """
    Plot PC scores over time for all legs in a given sequence.
    
    Parameters
    ----------
    scores_df : pandas.DataFrame
        DataFrame containing scores for all legs with columns:
        'sequenceID', 'time_in_frames', 'leg_number', 'PC1', 'PC2', etc.
    sequence_id : str
        Sequence ID to plot
    components_list : list, optional
        List of principal components to plot (default: None)
    leg_list : list, optional
        List of legs to plot (default: None)
    figsize : tuple, optional
        Figure size in inches (default: (8, 4))
        
    Returns
    -------
    tuple
        matplotlib figure and axes objects
    """
    # Color scheme for different legs
    colourList = ["#FC9CF9", "#73E6E2", "#ADD487", "#5F62BF",
                  "#C5C6E8", "#d6e9c3", "#BAF3F1", "#fdc4fb"]
    
    # Create subplot grid
    if components_list is None:
        components_list = range(4)
    else:
        components_list = [component-1 for component in components_list]


    if leg_list is None:
        leg_list = range(8)
    else:
        leg_list = [leg-1 for leg in leg_list]

    n_components = len(components_list)
    n_cols = 1
    n_rows = n_components
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
    
    if n_components > 1:
        ax = ax.flatten()
    else:
        ax = [ax]
    
    # Get data for the specified sequence
    current_seq = scores_df[scores_df["sequenceID"] == sequence_id]
    
    # Plot each leg's scores
    for leg_idx in leg_list:  # Assuming 8 legs
        leg_data = current_seq[current_seq["leg_number"] == leg_idx + 1]
        
        for pc_idx in components_list:
            ax[pc_idx].plot(leg_data["time_in_frames"], 
                          leg_data[f"PC{pc_idx+1}"],
                          label=f"Leg {leg_idx+1}",
                          linewidth=1,
                          color=colourList[leg_idx])
            ax[pc_idx].set_title(f"PC {pc_idx+1}")
            ax[pc_idx].set_xlabel("Time in Frames")
            ax[pc_idx].set_ylabel(f"PC {pc_idx+1} Score")
    
    # Add legend to the last subplot
    ax[-1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    return fig, ax