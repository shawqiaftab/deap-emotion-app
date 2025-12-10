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
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
    
    if isinstance(checkpoint, dict) and not hasattr(checkpoint, 'eval'):
        st.warning(f"⚠️ {Path(model_path).name} is a state dict. Use Keras models instead or define model architecture.")
        return None
    else:
        checkpoint.eval()
        return checkpoint

@st.cache_resource
def load_all_fold_models(model_dir, extensions=None):
    if extensions is None:
        extensions = ['.h5']
    
    models = {}
    model_path = Path(model_dir)
    
    if not model_path.exists():
        st.warning(f"Model directory not found: {model_dir}")
        return models
    
    for ext in extensions:
        for path in sorted(model_path.glob(f'*{ext}')):
            fold_name = path.stem
            if ext == '.h5':
                models[fold_name] = load_keras_model(str(path))
            elif ext in ['.pt', '.pth']:
                loaded = load_pytorch_model(str(path))
                if loaded is not None:
                    models[fold_name] = loaded
    
    return models

