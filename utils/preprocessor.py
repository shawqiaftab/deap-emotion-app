import numpy as np
import cv2
import h5py
import streamlit as st
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

def load_eeg_data(eeg_file):
    data = np.load(eeg_file)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data

def load_video_data(video_file, num_frames=20, target_size=(224, 224)):
    temp_path = f"/tmp/{video_file.name}"
    with open(temp_path, 'wb') as f:
        f.write(video_file.read())
    
    cap = cv2.VideoCapture(temp_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, target_size)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

@st.cache_resource
def get_resnet_model():
    return ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_video_embeddings(video_frames):
    model = get_resnet_model()
    preprocessed = preprocess_input(video_frames.astype(np.float32))
    embeddings = model.predict(preprocessed, verbose=0)
    return embeddings.mean(axis=0, keepdims=True)

def load_fused_data(h5_file, subject, trial):
    temp_path = f"/tmp/{h5_file.name}"
    with open(temp_path, 'wb') as f:
        f.write(h5_file.read())
    
    with h5py.File(temp_path, 'r') as f:
        path = f"s{subject:02d}/trial_{trial:02d}"
        return {
            'eeg': f[f"{path}/raw_eeg"][:],
            'peripheral': f[f"{path}/raw_peripheral"][:],
            'video_embedding': f[f"{path}/video_embedding"][:],
            'labels': f[f"{path}/labels"][:]
        }

def validate_eeg_shape(eeg_data, expected_shape=(32, 8064)):
    if eeg_data.shape != expected_shape:
        raise ValueError(f"Expected EEG shape {expected_shape}, got {eeg_data.shape}")
    return True

def validate_video_embedding_shape(video_emb, expected_shape=(768,)):
    if video_emb.shape[-1] != expected_shape[0]:
        raise ValueError(f"Expected video embedding dim {expected_shape[0]}, got {video_emb.shape}")
    return True

