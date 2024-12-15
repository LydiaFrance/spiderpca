import numpy as np
import pandas as pd


def load_and_process_spider_data(file_path:str, 
                                 species:str = None ,
                                 exclude_center:bool = True,
                                 remove_nan:bool = True,
                                 rescale_metres:bool = True) -> pd.DataFrame:
    """
    Load and process spider data from a CSV file.
    Steps:
    - Load data
    - Filter by species (optional)
    - Remove nan values (optional)
    - Reshape to 3D
    - Rescale to metres (optional)
    - Return marker data

    Inputs:
        file_path: str, path to the CSV file containing spider data
        species: str, optional, species to filter by (default: None)

    Returns:
        numpy array of marker coordinates [n_frames, n_markers, 3]
        DataFrame of spider data with nan values removed (optional)
    """
    spider_data = load_spider_data(file_path, species)
    marker_columns = get_marker_columns(spider_data, exclude_center=exclude_center)
    marker_data, marker_columns, spider_data_df = get_marker_data(spider_data, 
                                                  marker_columns, 
                                                  remove_nan=remove_nan, 
                                                  rescale_metres=rescale_metres)
    
    return marker_data, marker_columns, spider_data_df

# ------------------------- HELPER FUNCTIONS -----------------------------

def load_spider_data(file_path:str, 
                     species:str = None) -> pd.DataFrame:
    """
    Load spider data from a CSV file and filter by species if specified.

    Input:
        file_path: str, path to the CSV file containing spider data
        species: str, optional, species to filter by (default: None)

    Returns:
        pd.DataFrame, filtered spider data  
    """
    spider_data = pd.read_csv(file_path)
    if species is not None:
        spider_data = spider_data[spider_data["species"] == species]
        print(f"Filtered for {species} spider data.")
    return spider_data


def get_marker_columns(data: pd.DataFrame, 
                       exclude_center:bool = True) -> list:
    """
    Get column names from spider dataframe for marker coordinates.
    
    Inputs:
        data: DataFrame containing spider data
    
    Returns:
        List of column names ending in _x, _y, or _z, excluding those with 'center'
    """
    marker_columns = [col for col in data.columns 
              if col.endswith(("_x", "_y", "_z")) 
              and "center" not in col]

    if exclude_center:
        marker_columns = [col for col in marker_columns if "center" not in col]
    return marker_columns 

def get_marker_data(spider_data_df: pd.DataFrame, 
                     marker_columns: list, 
                     remove_nan: bool = True,
                     rescale_metres: bool = True) -> np.ndarray:
    """
    Get marker data from spider dataframe.
    
    Inputs:
        spider_data: DataFrame containing spider data
        marker_columns: list of column names to extract
        remove_nan: bool, whether to remove nan values (default: True)
        rescale_metres: bool, whether to rescale to metres (default: True)
    
    Returns:
        numpy array of marker coordinates [n_frames, n_markers, 3]
        DataFrame of spider data with nan values removed (optional)
    """
    
    # Remove nan values
    if remove_nan:
        length_before = spider_data_df.shape[0]
        spider_data_df = spider_data_df.dropna()
        length_after = spider_data_df.shape[0]
        if length_after < length_before:
            print(
                f"{length_before - length_after}" 
                f" rows with NaN values were removed."
                f" Now {length_after} rows."
            )

    # Reshape to 3D
    marker_data = spider_data_df[marker_columns].to_numpy()
    marker_data = marker_data.reshape(spider_data_df.shape[0], -1, 3)

    # Rescale to metres
    if rescale_metres:
        marker_data = marker_data / 1000
        print("Marker data rescaled to metres.")

    return marker_data, marker_columns, spider_data_df


