import numpy as np
from sklearn.decomposition import PCA

def run_PCA(markers, project_data=None):
    """
    Run Principal Component Analysis on the given markers data.

    Args:
        markers (np.ndarray): Input marker data.
        project_data (np.ndarray, optional): Additional data to project onto the PCA space.

    Returns:
        Tuple[np.ndarray, np.ndarray, PCA]: Principal components, scores, and PCA object.

    Raises:
        ValueError: If the input data shapes are inconsistent.
    """
    # Reshape the data to be [n, nMarkers*3]
    pca_input = get_PCA_input(markers)

    # Run PCA
    pca = PCA()
    pca_output = pca.fit(pca_input)

    # User may want to fit the principle components 
    # to a different dataset
    if project_data is None:
        project_data = pca_input
    else:
        project_data = get_PCA_input(project_data)

    # Another word for eigenvectors is components.
    principal_components = pca_output.components_
    
    # Another word for scores is projections.
    scores = pca_output.transform(project_data)

    # Check the shape of the output
    try:
        test_PCA_output(project_data, principal_components, scores)
    except AssertionError as msg:
        raise ValueError(f"PCA output validation failed: {str(msg)}")

    return principal_components, scores, pca

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def get_PCA_input_sizes(pca_input):
    """
    Get the sizes of the input data.
    """
    
    n_frames = pca_input.shape[0]
    n_markers = pca_input.shape[1]/3
    n_vars = pca_input.shape[1]

    return n_frames, n_markers, n_vars

def get_PCA_input(markers):
    """
    Reshape the data to be [n, nMarkers*3]
    """
    n_markers = markers.shape[1]
    pca_input = markers.reshape(-1, n_markers*3)

    return pca_input


def test_PCA_output(pca_input, principal_components, scores):
    """
    Test the shape of the PCA output.
    """
    n_frames, n_markers, n_vars = get_PCA_input_sizes(pca_input)

    assert n_vars == n_markers*3, "n_vars is not equal to n_markers*3."
    assert principal_components.shape[0] == n_vars, "principal_components is not the right shape."
    assert principal_components.shape[1] == n_vars, "principal_components is not the right shape."
    assert scores.shape[0] == n_frames, "scores first dim is not the right shape."
    assert scores.shape[1] == n_vars, "scores second dim is not the right shape."

