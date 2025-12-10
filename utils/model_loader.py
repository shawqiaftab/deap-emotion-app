import streamlit as st
import keras
import torch
from pathlib import Path

@st.cache_resource
def load_keras_model(model_path):
    custom_objects = {
        'mse': keras.losses.MeanSquaredError(),
        'mae': keras.losses.MeanAbsoluteError(),
        'accuracy': keras.metrics.BinaryAccuracy()
    }
    return keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)

@st.cache_resource
def load_pytorch_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval()
    return model

@st.cache_resource
def load_all_fold_models(model_dir, extension='.h5'):
    models = {}
    model_path = Path(model_dir)
    
    if not model_path.exists():
        st.warning(f"Model directory not found: {model_dir}")
        return models
    
    for path in sorted(model_path.glob(f'*{extension}')):
        fold_name = path.stem
        if extension == '.h5':
            models[fold_name] = load_keras_model(str(path))
        elif extension in ['.pt', '.pth']:
            models[fold_name] = load_pytorch_model(str(path))
    
    return models

