import plotly.graph_objects as go
import numpy as np

def plot_leg_overlay(new_legs, num_samples=100, title="Overlay of Random Frames for All Aligned Legs (Interactive 3D)"):
    """
    Create an interactive 3D plot showing random frames of spider leg movements.
    
    Args:
        new_legs (np.ndarray): Array of leg positions with shape (frames, legs, points, coordinates)
        num_samples (int): Number of random frames to sample (default: 100)
        title (str): Plot title
    
    Returns:
        go.Figure: The plotly figure object
    """
    # Create a new figure
    fig = go.Figure()

    # Define a color palette for the legs
    # colors = ["orange", "blue", 
    #           "green", "purple", 
    #           "pink", "lightgreen", 
    #           "lightblue", "yellow"]
    
    colourList = ["#e84855", "#FF9B71", "#FFFD82", "#1B998B",
                  "#86EADE","#FFFEC2","#FFB899","#F2929A"]

    # Number of legs
    num_legs = new_legs.shape[1]

    # Randomly select frame indices from the available frames
    total_frames = new_legs.shape[0]
    random_frames = np.random.choice(total_frames, size=min(num_samples, total_frames), replace=False)

    for leg_idx in range(num_legs):
        color = colourList[leg_idx % len(colourList)]

        # Iterate over the randomly selected frames for the current leg
        for frame_idx in random_frames:
            aligned_scatter_leg = go.Scatter3d(
                x=new_legs[frame_idx, leg_idx, :, 0],
                y=new_legs[frame_idx, leg_idx, :, 1],
                z=new_legs[frame_idx, leg_idx, :, 2],
                mode="markers+lines",
                marker=dict(size=2, color=color, opacity=0.5),
                line=dict(color=color, width=1),
                name=f"Aligned Leg {leg_idx + 1} - Frame {frame_idx + 1}"
            )
            fig.add_trace(aligned_scatter_leg)

            # Make the end of the leg black x marker
            end_scatter = go.Scatter3d(
                x=[new_legs[frame_idx, leg_idx, 0, 0]],
                y=[new_legs[frame_idx, leg_idx, 0, 1]],
                z=[new_legs[frame_idx, leg_idx, 0, 2]],
                mode="markers",
                marker=dict(size=1, color="black"),
            )
            fig.add_trace(end_scatter)

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
        ),
        title=title,
        showlegend=False  # Hide legend to avoid clutter
    )

    return fig