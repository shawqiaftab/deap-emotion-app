import numpy as np
import pandas as pd
import pickle
import h5py
import streamlit as st

def load_eeg_data(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1]
    
    if file_extension == 'npy':
        data = np.load(uploaded_file)
    elif file_extension == 'csv':
        data = pd.read_csv(uploaded_file).values
    elif file_extension == 'pkl':
        data = pickle.load(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    if len(data.shape) == 2 and data.shape[0] == 32:
        data = np.expand_dims(data, axis=0)
    
    return data

def load_video_data(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1]
    
    if file_extension == 'npy':
        data = np.load(uploaded_file)
    elif file_extension == 'pkl':
        data = pickle.load(uploaded_file)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")
    
    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=0)
    
    return data

def load_fused_data(uploaded_file):
    with h5py.File(uploaded_file, 'r') as f:
        subject = list(f.keys())[0]
        trial = list(f[subject].keys())[0]
        
        eeg = f[f'{subject}/{trial}/raw_eeg'][:]
        video = f[f'{subject}/{trial}/video_embedding'][:]
        labels = f[f'{subject}/{trial}/labels'][:]
        
        eeg = np.expand_dims(eeg, axis=0)
        video = np.expand_dims(video, axis=0)
    
    return eeg, video, labels

def validate_eeg_shape(data):
    expected_channels = 32
    expected_samples = 8064
    
    if data.shape[1] == expected_channels and data.shape[2] == expected_samples:
        return data
    else:
        st.warning(f"Expected EEG shape: (batch, 32, 8064), got: {data.shape}")
        return data

def validate_video_shape(data):
    expected_embedding_dim = 768
    
    if len(data.shape) == 2 and data.shape[1] == expected_embedding_dim:
        return data
    else:
        st.warning(f"Expected video embedding shape: (batch, 768), got: {data.shape}")
        return data

