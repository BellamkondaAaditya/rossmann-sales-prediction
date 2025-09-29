"""
Utility functions for Rossmann Store Sales Prediction project
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json


def setup_logging(log_file='logs/pipeline.log', level=logging.INFO):
    """
    Setup logging configuration
    
    Parameters:
    -----------
    log_file : str
        Path to log file
    level : logging level
        Logging level (default: INFO)
    """
    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def create_directories(dirs):
    """
    Create directories if they don't exist
    
    Parameters:
    -----------
    dirs : list
        List of directory paths to create
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        

def save_dataframe(df, filepath, logger=None):
    """
    Save dataframe to CSV with logging
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to save
    filepath : str
        Path to save file
    logger : logging.Logger
        Logger instance
    """
    df.to_csv(filepath, index=False)
    
    if logger:
        logger.info(f"Saved dataframe to {filepath}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Size: {os.path.getsize(filepath) / 1024**2:.2f} MB")


def load_dataframe(filepath, parse_dates=None, logger=None):
    """
    Load dataframe from CSV with logging
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    parse_dates : list
        Columns to parse as dates
    logger : logging.Logger
        Logger instance
        
    Returns:
    --------
    pd.DataFrame
    """
    if logger:
        logger.info(f"Loading data from {filepath}")
    
    df = pd.read_csv(filepath, parse_dates=parse_dates, low_memory=False)
    
    if logger:
        logger.info(f"Loaded dataframe: {df.shape}")
    
    return df


def print_section_header(title, char='=', width=80):
    """
    Print a formatted section header
    
    Parameters:
    -----------
    title : str
        Section title
    char : str
        Character to use for border
    width : int
        Width of header
    """
    print('\n' + char * width)
    print(f' {title} '.center(width, char))
    print(char * width + '\n')


def calculate_missing_stats(df):
    """
    Calculate missing value statistics
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    pd.DataFrame
        Missing value statistics
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Missing_Count': missing,
        'Percentage': missing_pct
    }).sort_values('Missing_Count', ascending=False)
    
    return missing_df[missing_df['Missing_Count'] > 0]


def save_plot(fig, filename, output_dir='reports/figures', dpi=300):
    """
    Save matplotlib figure
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Filename (without path)
    output_dir : str
        Output directory
    dpi : int
        Resolution
    """
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def calculate_rmspe(y_true, y_pred):
    """
    Calculate Root Mean Square Percentage Error
    
    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        RMSPE score
    """
    # Avoid division by zero
    mask = y_true != 0
    return np.sqrt(np.mean(((y_true[mask] - y_pred[mask]) / y_true[mask]) ** 2))


def memory_usage_mb(df):
    """
    Calculate dataframe memory usage in MB
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    float
        Memory usage in MB
    """
    return df.memory_usage(deep=True).sum() / 1024**2


def reduce_memory_usage(df, verbose=True):
    """
    Reduce memory usage by optimizing dtypes
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    verbose : bool
        Print memory reduction info
        
    Returns:
    --------
    pd.DataFrame
        Optimized dataframe
    """
    start_mem = memory_usage_mb(df)
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = memory_usage_mb(df)
    
    if verbose:
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB '
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    
    return df


def save_json(data, filepath):
    """
    Save dictionary to JSON file
    
    Parameters:
    -----------
    data : dict
        Data to save
    filepath : str
        Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(filepath):
    """
    Load JSON file
    
    Parameters:
    -----------
    filepath : str
        Input file path
        
    Returns:
    --------
    dict
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def get_categorical_columns(df):
    """
    Get list of categorical columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    list
        List of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()


def get_numerical_columns(df):
    """
    Get list of numerical columns
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
        
    Returns:
    --------
    list
        List of numerical column names
    """
    return df.select_dtypes(include=[np.number]).columns.tolist()
