import numpy as np


def reconstruct(score_frames, principal_components, mu, components_list=None):
    """
    Reconstruct frames from PCA components and scores by projecting back to the original space.

    Parameters
    ----------
    score_frames : numpy.ndarray, shape (n_frames, n_components)
        The projection of the original data onto the principal components (PC scores).
    principal_components : numpy.ndarray, shape (n_components, n_markers * 3)
        The principal components matrix representing the basis vectors.
    mu : numpy.ndarray, shape (1, n_markers, 3)
        The mean pose that was subtracted during PCA computation.
    components_list : list or None, optional
        Indices of components to use for reconstruction. If None, all components are used.
        Default is None.

    Returns
    -------
    numpy.ndarray, shape (n_frames, n_markers, 3)
        The reconstructed frames in the original space.

    Raises
    ------
    TypeError
        If score_frames is not a numpy array.
    ValueError
        If score_frames is not 2-dimensional.
        If dimensions of inputs don't match.

    """

    if components_list is None:
        components_list = range(principal_components.shape[1])

    if not isinstance(score_frames, np.ndarray):
        raise TypeError("score_frames must be a numpy array.")

    if len(score_frames.shape) != 2:
        raise ValueError("score_frames must be 2d.")
    
    assert score_frames.shape[1] == principal_components.shape[0], "score_frames must have the same number of columns as components_list."
    assert len(components_list) <= principal_components.shape[1], "components_list must not exceed the number of principal components."
    assert len(mu.shape)==3, "mu must be a 3d array: [1,nMarkers,3]."

    n_markers = mu.shape[1]
    n_dims = mu.shape[2]
    n_frames = score_frames.shape[0]

    # Select principal components and scores based on the provided list
    selected_PCs = principal_components[components_list,:] # principal_components is [n_components, n_markers*3]
    selected_scores = score_frames[:, components_list] # score_frames is [n_frames, n_components]

    reconstruction = np.dot(selected_scores,selected_PCs) # [n_frames, n_markers*3]
    reconstruction = reconstruction.reshape(-1, n_markers, n_dims)  # Reshape to [n_frames, n_markers, 3]

    reconstructed_frames = mu + reconstruction  # Broadcasting [1, n_markers, 3] over [n_frames, n_markers, 3]

    assert reconstructed_frames.shape[0] == n_frames, "Reconstructed frames do not match the number of frames."
    assert reconstructed_frames.shape[1] == n_markers, "Reconstructed frames do not match the number of markers."
    assert reconstructed_frames.shape[2] == n_dims, "Reconstructed frames do not match the number of dimensions."

    return reconstructed_frames

