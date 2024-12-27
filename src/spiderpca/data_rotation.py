import numpy as np


def undo_body_rotation(markers, whole_body_angle, degrees=True, which_axis='z'):

    """
    Undo the rotation of the spider's body.

    Inputs:
        markers: np.ndarray, shape (n_frames, n_markers, 3)
        whole_body_angle: in degrees,np.ndarray, shape (n_frames,1)
        which_axis: str, the axis of rotation ('x', 'y', 'z')

    Returns:
        np.ndarray, shape (n_frames, n_markers, 3)
    """

    # Check that the input arrays have the correct shapes
    if markers.shape[0] != whole_body_angle.shape[0]:
        raise ValueError("The number of frames in markers and whole_body_angle must match.")
    
    if degrees:     
        # Convert body pitch angles from degrees to radians
        body_pitch_rad = np.radians(whole_body_angle)
    else:
        body_pitch_rad = whole_body_angle
    
    # Prepare an array to hold the corrected markers
    n_frames = markers.shape[0]
    corrected_markers = np.empty_like(markers)
    
    # Iterate over each instance
    for i in range(n_frames):
        # Get the current pitch angle
        pitch = body_pitch_rad[i]
        
        if which_axis == 'z':
            # Compute the rotation matrix to undo the pitch rotation
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)]
            ])
        elif which_axis == 'x':
            rotation_matrix = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
        elif which_axis == 'y':
            rotation_matrix = np.array([
                [np.cos(pitch), -np.sin(pitch), 0],
                [np.sin(pitch), np.cos(pitch), 0],
                [0, 0, 1]
            ])
        else:
            print(f"Invalid axis: {which_axis}")
            return markers
        
        # Apply the inverse rotation to each marker (by applying the rotation matrix)
        corrected_markers[i] = markers[i] @ rotation_matrix.T
    
    return corrected_markers

