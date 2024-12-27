import numpy as np


def get_leg_markers(marker_names, markers, leg_id):
    """
    Extracts markers for a specific leg based on a naming pattern.
    
    Parameters:
        marker_names (list of str): List of all marker names.
        markers (ndarray): 3D array of shape (n_instances, n_markers, 3) containing marker positions.
        leg_id (int): Identifier for the leg (e.g., 1 for leg 1).
        
    Returns:
        ndarray: Extracted markers for the specified leg of shape (n_instances, n_leg_markers, 3).
        list of str: Names of the extracted markers.
    """

    # Make a copy of the markers to avoid modifying the original
    markers = markers.copy()

    leg_id = str(leg_id)

    # Filter marker names based on the leg identifier
    leg_markers_names = [name for name in marker_names if f"{leg_id}" in name]
    
    # Find the indices of the selected markers
    leg_markers_indices = [marker_names.index(name) for name in leg_markers_names]
    
    # Extract the corresponding markers from the dataset
    extracted_leg_markers = markers[:, leg_markers_indices, :]
    
    return extracted_leg_markers, leg_markers_names

def get_all_legs_markers(marker_names, markers, num_legs):
    """
    Extracts markers for all legs and organizes them into a unified numpy array.

    Parameters:
        marker_names (list of str): List of all marker names.
        markers (ndarray): 3D array of shape (n_instances, n_markers, 3).
        num_legs (int): Total number of legs.

    Returns:
        ndarray: Markers organized as [frames, leg, keypoints, dims].
        list of list of str: Names of markers for each leg.
    """


    # Make a copy of the markers to avoid modifying the original
    markers = markers.copy()

    all_legs = []
    all_legs_names = []
    
    for leg_id in range(1, num_legs + 1):  # Loop through each leg
        leg_markers, leg_markers_names = get_leg_markers(marker_names, markers, leg_id)
        all_legs.append(leg_markers)
        all_legs_names.append(leg_markers_names)
    
    # Stack into a single array with shape [frames, leg, keypoints, dims]
    all_legs = np.stack(all_legs, axis=1)

    # Make a matrix with original dimensions of  [frames, markers, dims]
    # This is just for testing
    # Search for index using marker names
    markers_again = np.zeros((markers.shape[0], markers.shape[1], markers.shape[2]))
    for index, name in enumerate(marker_names):
        markers_again[:,index, :] = markers[:, index, :]
    

    return all_legs, all_legs_names, markers_again



def put_legs_back(
    all_legs,
    all_legs_names,
    original_marker_names,
    spider3d_markers,
    original_markers = None,
):
    """
    Reconstructs the full markers dataset, combining aligned leg markers with the original non-leg markers.
    Ensures that the order of markers is consistent with original_marker_names.
    Optionally replaces non-leg markers with a single value from spider3d_markers.

    Parameters:
        all_legs (ndarray): Aligned leg markers of shape [nframes, nlegs, nmarkers, 3].
        all_legs_names (list of list of str): Names of markers for each leg.
        original_marker_names (list of str): Full list of marker names in the original dataset.
        original_markers (ndarray): Original markers array before alignment [nframes, nmarkers, 3].
        spider3d_markers (ndarray, optional): A single frame of markers [nmarkers, 3]. 
            If provided, all non-leg markers will be replaced with these values.

    Returns:
        ndarray: Full reconstructed markers array with shape [nframes, nmarkers, 3].
    """
    nframes = all_legs.shape[0]
    nmarkers = spider3d_markers.shape[1]
    ndims = all_legs.shape[3]

    all_legs = all_legs.copy()

    # Initialize reconstructed markers
    reconstructed_markers = np.zeros((nframes, nmarkers, ndims))

    # Flatten all leg marker names into a single list for quick lookup
    leg_marker_names = [name for sublist in all_legs_names for name in sublist]
    leg_marker_to_aligned_index = {}
    for leg_idx, leg_marker_names_per_leg in enumerate(all_legs_names):
        for marker_idx, marker_name in enumerate(leg_marker_names_per_leg):
            leg_marker_to_aligned_index[marker_name] = (leg_idx, marker_idx)

    # Fill in markers based on their source
    for marker_idx, marker_name in enumerate(original_marker_names):
        if marker_name in leg_marker_to_aligned_index:  # If the marker belongs to a leg
            leg_idx, leg_marker_idx = leg_marker_to_aligned_index[marker_name]
            reconstructed_markers[:, marker_idx, :] = all_legs[:, leg_idx, leg_marker_idx, :]
        elif original_markers is not None:  # Use spider3d_markers for non-leg markers
            reconstructed_markers[:, marker_idx, :] = original_markers[:, marker_idx, :]
        else:  # Default to original_markers if spider3d_markers is not provided
            reconstructed_markers[:, marker_idx, :] = spider3d_markers[:, marker_idx, :]

    # restore the coxa markers
    if original_markers is None:
        coxa_markers_names = [name for name in original_marker_names if "coxa" in name]
        coxa_markers_indices = [original_marker_names.index(name) for name in coxa_markers_names]
        coxa_markers = spider3d_markers[:, coxa_markers_indices, :]
        reconstructed_markers[:, coxa_markers_indices, :] = coxa_markers

    

    return reconstructed_markers
