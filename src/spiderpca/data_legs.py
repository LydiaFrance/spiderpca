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
    print(f"All legs shape: {all_legs.shape}")
    print(f"Frames: {all_legs.shape[0]}, Legs: {all_legs.shape[1]}, Keypoints: {all_legs.shape[2]}, Dims: {all_legs.shape[3]}")

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


def make_coxa_origin(all_legs):
    """
    Translates all points so that the coxa (4th keypoint) becomes the origin for each leg.
    
    Parameters:
        all_legs (ndarray): Array of shape [nFrames, nLegs, 4, 3] where the last two dimensions are
                          keypoints and xyz coordinates, and coxa is the 4th keypoint
    
    Returns:
        ndarray: Array of same shape with all points translated relative to coxa
        ndarray: Array of shape [nFrames, nLegs, 4, 3] containing the coxa positions
    """
    # Make a copy to avoid modifying the original
    all_legs = all_legs.copy()
    
    # Get the coxa positions (index -1 for last keypoint)
    coxa = all_legs[..., -1:, :]
    
    # Subtract coxa position from all points
    all_legs = all_legs - coxa
    
    return all_legs, coxa

def unmake_coxa_origin(all_legs, coxa):
    return all_legs + coxa



def reflect_legs(all_legs):
    """
    Reflects the left legs (5-8) to match right legs (1-4)

    Parameters:
        all_legs (ndarray): Array of shape [nframes, nlegs, nkeypoints, ndims]
    
    Returns:
        ndarray: Array of shape [nframes, nlegs, nkeypoints, ndims] with left legs reflected
    """

    # Reflect the left legs (y-coordinate)
    # Note: Python 0-based indexing, so legs 5-8 are indices 4-7


    all_legs = all_legs.copy()
    all_legs[:, 4:8, :, 1] = -all_legs[:, 4:8, :, 1]

    return all_legs

def combine_legs(all_legs):
    """
    Reflects the left legs (5-8) to match right legs (1-4) and combines them.
    
    Parameters:
        all_legs (ndarray): Array of shape [nframes, nlegs, nkeypoints, ndims]
    
    Returns:
        ndarray: Array of shape [nframes*2, nlegs//2, nkeypoints, ndims] where frames 
                dimension now includes both original and reflected legs
    """
    # Make a copy to avoid modifying the original
    all_legs = all_legs.copy()
    
    # Reshape to combine corresponding left-right pairs
    nframes, _, nkeypoints, ndims = all_legs.shape
    # Separate into left and right legs and stack them as new frames
    right_legs = all_legs[:, :4]  # legs 1-4
    left_legs = all_legs[:, 4:]   # legs 5-8
    
    # Stack right and left legs along the frames dimension
    combined_legs = np.concatenate([right_legs, left_legs], axis=0)
    
    print(f"Combined legs shape: {combined_legs.shape}")  # Should be (nframes*2, 4, 4, 3)
    return combined_legs


def restore_leg_positions(reconstructed_frames, spider3d, all_legs_names):
    """
    Transform reconstructed leg movements back to original coordinate space.
    
    Parameters
    ----------
    reconstructed_frames : ndarray
        Reconstructed frames from PCA, shape (n_frames, n_markers, 3)
    spider3d : Spider3D
        Spider3D object containing marker information
    nLegs : int
        Number of legs
    all_legs_names : list
        List of marker names for each leg
    
    Returns
    -------
    ndarray
        Reconstructed markers in original coordinate space
    """
    # Get coxa indices

    nLegs = 8

    coxa_names = [f"coxa{i}" for i in range(1, nLegs + 1)]
    coxa_indices = spider3d.skeleton_definition.get_marker_indices(coxa_names)
    
    # Add leg dimension and repeat for each leg
    reconstructed_frames = np.expand_dims(reconstructed_frames, axis=1)
    reconstructed_frames = np.repeat(reconstructed_frames, nLegs, axis=1)
    
    # Prepare coxa positions
    original_coxa_positions = spider3d.markers[:, coxa_indices, :]
    original_coxa_positions = np.repeat(original_coxa_positions, reconstructed_frames.shape[0], axis=0)
    original_coxa_positions = np.expand_dims(original_coxa_positions, axis=2)
    
    # Transform back to original coordinate space
    restored_legs = reflect_legs(reconstructed_frames)
    restored_legs = unmake_coxa_origin(restored_legs, original_coxa_positions)
    restored_legs = put_legs_back(
        all_legs=restored_legs,
        all_legs_names=all_legs_names,
        spider3d_markers=spider3d.markers.reshape(1, -1, 3),
        original_marker_names=spider3d.marker_names,
    )
    
    return restored_legs