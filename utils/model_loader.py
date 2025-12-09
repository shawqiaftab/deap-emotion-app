import streamlit as st
import torch
from tensorflow import keras
from pathlib import Path

@st.cache_resource
def load_keras_model(model_path):
    return keras.models.load_model(model_path)

@st.cache_resource
def load_pytorch_model(model_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

@st.cache_resource
def load_all_fold_models(model_dir, extension):
    models = []
    model_paths = sorted(Path(model_dir).glob(f'*.{extension}'))
    
    for path in model_paths:
        if extension == 'h5':
            model = load_keras_model(str(path))
        else:
            model = load_pytorch_model(str(path))
        models.append(model)
    
    return models

