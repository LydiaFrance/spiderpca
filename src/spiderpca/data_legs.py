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
