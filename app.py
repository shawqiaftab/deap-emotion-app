import streamlit as st
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from utils.model_loader import load_pytorch_model, load_keras_model

st.set_page_config(page_title="DEAP Emotion Detection", layout="wide")

st.title("🧠 DEAP Emotion Detection - Multi-Model Inference")

MODEL_DIR = Path("models")

def get_available_models():
    models = {}
    if not MODEL_DIR.exists():
        return models
    
    for folder in MODEL_DIR.iterdir():
        if folder.is_dir():
            model_files = list(folder.glob("*.h5")) + list(folder.glob("*.pt")) + list(folder.glob("*.pth"))
            if model_files:
                models[folder.name] = sorted([f.name for f in model_files])
    return models

def load_model(model_path):
    if model_path.endswith('.h5'):
        return load_keras_model(str(model_path))
    elif model_path.endswith(('.pt', '.pth')):
        return load_pytorch_model(str(model_path))
    else:
        raise ValueError(f"Unsupported format: {model_path}")

def predict_pytorch(model, eeg_features, video_features):
    eeg_tensor = torch.FloatTensor(eeg_features).flatten().unsqueeze(0)
    video_tensor = torch.FloatTensor(video_features).unsqueeze(0)
    
    with torch.no_grad():
        output = model(eeg_tensor, video_tensor)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1)
    
    return pred.item(), probs.numpy()[0]

def predict_keras(model, eeg_features, video_features):
    eeg_input = eeg_features.flatten().reshape(1, -1)
    video_input = video_features.reshape(1, -1)
    
    output = model.predict([eeg_input, video_input], verbose=0)
    pred = np.argmax(output, axis=1)[0]
    probs = output[0]
    
    return pred, probs

st.sidebar.header("Model Selection")

available_models = get_available_models()

if not available_models:
    st.error("No models found! Put models in 'models/' folder")
    st.stop()

selected_folder = st.sidebar.selectbox("Model Type", list(available_models.keys()))
selected_file = st.sidebar.selectbox("Model File", available_models[selected_folder])

model_path = MODEL_DIR / selected_folder / selected_file

if st.sidebar.button("Load Model"):
    with st.spinner("Loading model..."):
        try:
            model = load_model(model_path)
            st.session_state['model'] = model
            st.session_state['model_type'] = 'keras' if selected_file.endswith('.h5') else 'pytorch'
            st.sidebar.success(f"✓ Loaded {selected_file}")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

if 'model' not in st.session_state:
    st.info("👈 Select and load a model from the sidebar")
    st.stop()

st.subheader("Input Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**EEG Features (32 channels × 10 bands)**")
    eeg_input = st.text_area("Paste EEG features (comma-separated or JSON)", height=150)

with col2:
    st.markdown("**Video Features (768-dim)**")
    video_input = st.text_area("Paste Video features (comma-separated or JSON)", height=150)

if st.button("🔮 Predict Emotion", type="primary"):
    try:
        if eeg_input and video_input:
            try:
                eeg_features = np.array(json.loads(eeg_input))
            except:
                eeg_features = np.array([float(x.strip()) for x in eeg_input.replace('\n', ',').split(',') if x.strip()])
            
            try:
                video_features = np.array(json.loads(video_input))
            except:
                video_features = np.array([float(x.strip()) for x in video_input.replace('\n', ',').split(',') if x.strip()])
            
            if eeg_features.size == 320:
                eeg_features = eeg_features.reshape(32, 10)
            
            st.write(f"EEG shape: {eeg_features.shape}, Video shape: {video_features.shape}")
            
            model = st.session_state['model']
            model_type = st.session_state['model_type']
            
            if model_type == 'pytorch':
                pred, probs = predict_pytorch(model, eeg_features, video_features)
            else:
                pred, probs = predict_keras(model, eeg_features, video_features)
            
            st.success("Prediction Complete!")
            
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.metric("Predicted Class", f"Class {pred}")
                st.metric("Confidence", f"{probs[pred]*100:.2f}%")
            
            with result_col2:
                st.markdown("**Class Probabilities:**")
                prob_df = pd.DataFrame({
                    'Class': [f'Class {i}' for i in range(len(probs))],
                    'Probability': probs,
                    'Percentage': [f'{p*100:.2f}%' for p in probs]
                })
                st.dataframe(prob_df, hide_index=True)
            
            st.bar_chart(prob_df.set_index('Class')['Probability'])
        
        else:
            st.warning("Please provide both EEG and Video features")
    
    except Exception as e:
        st.error(f"Prediction error: {e}")
        import traceback
        st.code(traceback.format_exc())

st.markdown("---")
st.markdown("**Sample Input Format:**")
st.code("""
EEG: [0.1, 0.2, ..., 0.3]  # 320 values or 32x10 array
Video: [0.5, 0.6, ..., 0.7]  # 768 values
""")
