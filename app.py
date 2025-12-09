import streamlit as st
import numpy as np
import torch
from pathlib import Path
from utils.model_loader import load_keras_model, load_pytorch_model, load_all_fold_models
from utils.preprocessor import load_eeg_data, load_video_data, validate_eeg_shape, validate_video_shape

st.set_page_config(page_title="DEAP Emotion Detection", layout="wide")

st.title("DEAP Multimodal Emotion Detection")
st.write("Emotion recognition using EEG signals (32 channels, 8064 samples @ 128Hz) and video embeddings (768-dim)")

st.sidebar.header("Model Configuration")

model_type = st.sidebar.selectbox(
    "Model Type",
    ["Video Only", "EEG Only", "Early Fusion", "Mid Fusion", "Late Fusion"]
)

validation = st.sidebar.selectbox(
    "Validation Strategy",
    ["Within-Subject", "LOSO"]
)

emotion_dim = None
if model_type == "Early Fusion" and validation == "Within-Subject":
    emotion_dim = st.sidebar.selectbox(
        "Emotion Dimension",
        ["arousal", "valence", "dominance", "liking"]
    )

st.subheader(f"Selected: {model_type} ({validation})")

model_configs = {
    ("Video Only", "Within-Subject"): ("models/video_only/final_model.h5", "single", "h5"),
    ("Video Only", "LOSO"): ("models/video_loso/", "multi", "h5"),
    ("EEG Only", "Within-Subject"): ("models/eeg_only/", "multi", "pt"),
    ("EEG Only", "LOSO"): ("models/eeg_loso/", "multi", "pt"),
    ("Early Fusion", "Within-Subject"): ("models/early_fusion/", "emotion_dim", "pt"),
    ("Early Fusion", "LOSO"): ("models/early_fusion_loso/", "multi", "pt"),
    ("Mid Fusion", "Within-Subject"): ("models/mid_fusion/", "multi", "pt"),
    ("Mid Fusion", "LOSO"): ("models/mid_fusion_loso/", "multi", "pt"),
    ("Late Fusion", "Within-Subject"): ("models/late_fusion/", "multi", "pth"),
    ("Late Fusion", "LOSO"): ("models/late_fusion_loso/", "multi", "pth"),
}

config = model_configs.get((model_type, validation))
if config:
    model_path, model_mode, extension = config
    if model_mode == "single":
        st.info(f"Using single model: {model_path}")
    elif model_mode == "emotion_dim":
        st.info(f"Using emotion-specific models from: {model_path}")
    else:
        if Path(model_path).exists():
            model_files = list(Path(model_path).glob(f"*.{extension}"))
            st.info(f"Using {len(model_files)} fold models for ensemble prediction")
        else:
            st.warning(f"Model directory not found: {model_path}")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("EEG Data Upload")
    if model_type in ["EEG Only", "Early Fusion", "Mid Fusion", "Late Fusion"]:
        st.caption("Expected shape: (32, 8064) - 32 channels, 8064 samples")
        eeg_file = st.file_uploader(
            "Upload EEG data (.npy, .csv, .pkl)",
            type=['npy', 'csv', 'pkl'],
            key='eeg'
        )
        if eeg_file:
            st.success(f"Uploaded: {eeg_file.name}")
    else:
        st.info("Not required for Video Only model")

with col2:
    st.subheader("Video Data Upload")
    if model_type in ["Video Only", "Early Fusion", "Mid Fusion", "Late Fusion"]:
        st.caption("Expected shape: (768,) - 768-dim embedding vector")
        video_file = st.file_uploader(
            "Upload Video features (.npy, .pkl)",
            type=['npy', 'pkl'],
            key='video'
        )
        if video_file:
            st.success(f"Uploaded: {video_file.name}")
    else:
        st.info("Not required for EEG Only model")

st.markdown("---")

predict_button = st.button("Predict Emotion", type="primary", use_container_width=True)

if predict_button:
    eeg_data = None
    video_data = None
    
    try:
        if model_type in ["EEG Only", "Early Fusion", "Mid Fusion", "Late Fusion"]:
            if eeg_file is None:
                st.error("Please upload EEG data")
                st.stop()
            eeg_data = load_eeg_data(eeg_file)
            eeg_data = validate_eeg_shape(eeg_data)
            st.success(f"EEG data loaded: Shape {eeg_data.shape}")
        
        if model_type in ["Video Only", "Early Fusion", "Mid Fusion", "Late Fusion"]:
            if video_file is None:
                st.error("Please upload Video data")
                st.stop()
            video_data = load_video_data(video_file)
            video_data = validate_video_shape(video_data)
            st.success(f"Video data loaded: Shape {video_data.shape}")
        
        with st.spinner("Loading model and predicting..."):
            model_path, model_mode, extension = config
            
            if model_mode == "single":
                model = load_keras_model(model_path)
                prediction = model.predict(video_data, verbose=0)
            
            elif model_mode == "emotion_dim":
                model_file = f"{model_path}{emotion_dim}.pt"
                model = load_pytorch_model(model_file)
                combined_data = np.concatenate([eeg_data.reshape(1, -1), video_data], axis=-1)
                input_tensor = torch.from_numpy(combined_data).float()
                with torch.no_grad():
                    prediction = model(input_tensor).cpu().numpy()
            
            else:
                models = load_all_fold_models(model_path, extension)
                predictions = []
                
                for model in models:
                    if extension == "h5":
                        pred = model.predict(video_data, verbose=0)
                    else:
                        if model_type == "EEG Only":
                            input_data = eeg_data
                        elif model_type in ["Early Fusion", "Mid Fusion"]:
                            eeg_flat = eeg_data.reshape(eeg_data.shape[0], -1)
                            input_data = np.concatenate([eeg_flat, video_data], axis=-1)
                        elif model_type == "Late Fusion":
                            input_data = eeg_data
                        else:
                            input_data = eeg_data
                        
                        input_tensor = torch.from_numpy(input_data).float()
                        with torch.no_grad():
                            pred = model(input_tensor).cpu().numpy()
                    
                    predictions.append(pred)
                
                prediction = np.mean(predictions, axis=0)
            
            st.success("Prediction Complete")
            
            st.subheader("Emotion Prediction Results")
            
            if model_mode == "emotion_dim":
                st.metric(emotion_dim.capitalize(), f"{prediction[0][0]:.4f}")
            else:
                col1, col2, col3, col4 = st.columns(4)
                emotions = ["Valence", "Arousal", "Dominance", "Liking"]
                
                for i, (col, emotion) in enumerate(zip([col1, col2, col3, col4], emotions)):
                    if prediction.shape[-1] > i:
                        value = prediction[0][i]
                        col.metric(emotion, f"{value:.4f}")
            
            with st.expander("View Raw Prediction"):
                st.json({
                    "prediction_shape": str(prediction.shape),
                    "prediction_values": prediction.tolist()
                })
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.exception(e)

