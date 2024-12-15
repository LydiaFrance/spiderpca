import numpy as np
import pandas as pd

def get_score_range(scores, num_frames=30):
    """
    Generate a series of scores for animations within a specified range.
    
    Parameters:
    - scores (numpy.ndarray): Array containing score values from which ranges are derived.
    - num_frames (int): Number of frames to generate scores for.
    
    Returns:
    - numpy.ndarray: An array of score values over the specified frame range.
    """

    min_score = np.mean(scores, axis=0) - (2 * np.std(scores, axis=0))
    max_score = np.mean(scores, axis=0) + (2 * np.std(scores, axis=0))

    # Create a triangle wave for the time series
    half_length = num_frames // 2 + 1
    triangle_wave = np.linspace(0, 1, half_length)
    triangle_wave = np.concatenate([triangle_wave, triangle_wave[-2:0:-1]])

    score_frames = min_score + (max_score - min_score) * triangle_wave[:, np.newaxis]

    return score_frames


def create_scores_dataframe(scores, spider_data_df, time_column='time_in_frames', filename_column='filename', sq_level_column='sq_level'):
    """
    Create a DataFrame containing PCA scores and metadata.
    
    Parameters
    ----------
    scores : numpy.ndarray
        PCA scores array
    metadata_df : pandas.DataFrame
        DataFrame containing metadata (must include time, filename, and sq_level columns)
    time_column : str, optional
        Name of the column containing time information (default: 'time_in_frames')
    filename_column : str, optional
        Name of the column containing filename/sequence IDs (default: 'filename')
    sq_level_column : str, optional
        Name of the column containing sq_level information (default: 'sq_level')
    
    Returns
    -------
    pandas.DataFrame
        DataFrame containing PCA scores and metadata
    """
    # Create DataFrame with PC scores
    scores_df = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(scores.shape[1])])
    
    # Add metadata columns
    scores_df["sq_level"] = spider_data_df[sq_level_column].to_numpy()
    scores_df["sequenceID"] = spider_data_df[filename_column]
    scores_df["time_in_frames"] = spider_data_df[time_column].to_numpy()
    
    return scores_df