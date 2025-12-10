import streamlit as st
import numpy as np
import torch
from pathlib import Path
from utils.model_loader import load_keras_model, load_pytorch_model, load_all_fold_models
from utils.preprocessor import (
    load_eeg_data, load_video_data, extract_video_embeddings, 
    load_fused_data, validate_eeg_shape, validate_video_embedding_shape
)

st.set_page_config(page_title="DEAP Emotion Detection", layout="wide")

st.title("🧠 DEAP Emotion Detection System")
st.markdown("Upload EEG and video data for multimodal emotion recognition")

with st.expander("📖 Understanding Emotion Dimensions"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Valence** (1-9)
        - Measures emotional positivity/negativity
        - Low (1-3): Negative emotions (sad, angry, fearful)
        - Medium (4-6): Neutral
        - High (7-9): Positive emotions (happy, joyful, excited)
        
        **Arousal** (1-9)
        - Measures emotional intensity/activation
        - Low (1-3): Calm, relaxed, sleepy
        - Medium (4-6): Alert, attentive
        - High (7-9): Excited, stimulated, energized
        """)
    
    with col2:
        st.markdown("""
        **Dominance** (1-9)
        - Measures sense of control/power
        - Low (1-3): Submissive, controlled, guided
        - Medium (4-6): Balanced influence
        - High (7-9): Dominant, in-control, influential
        
        **Liking** (1-9)
        - Measures preference/enjoyment
        - Low (1-3): Dislike, aversion
        - Medium (4-6): Indifferent
        - High (7-9): Like, preference, enjoyment
        """)

available_models = [
    "early_fusion",
    "early_fusion_loso", 
    "eeg_loso",
    "eeg_only",
    "late_fusion",
    "Late Fusion Loso",
    "mid_fusion",
    "mid_fusion_loso",
    "video_loso",
    "video_only"
]

st.sidebar.header("🤖 Model Selection")
selected_model_dir = st.sidebar.selectbox(
    "Choose Model Type",
    available_models,
    help="Select which trained model to use for predictions"
)

MODEL_DIR = Path("models") / selected_model_dir
st.sidebar.success(f"Using: `{selected_model_dir}`")

def generate_synthetic_eeg(channels=32, samples=8064, emotion_type='neutral'):
    duration = samples / 128
    time = np.linspace(0, duration, samples)
    eeg_data = np.zeros((channels, samples))
    
    for ch in range(channels):
        delta = 2.0 * np.sin(2 * np.pi * np.random.uniform(0.5, 4) * time)
        theta = 1.5 * np.sin(2 * np.pi * np.random.uniform(4, 8) * time)
        alpha = 3.0 * np.sin(2 * np.pi * np.random.uniform(8, 13) * time)
        beta = 1.0 * np.sin(2 * np.pi * np.random.uniform(13, 30) * time)
        gamma = 0.5 * np.sin(2 * np.pi * np.random.uniform(30, 50) * time)
        
        eeg_data[ch] = delta + theta + alpha + beta + gamma
        eeg_data[ch] += np.random.normal(0, 0.5, samples)
    
    if emotion_type == 'happy':
        eeg_data[8:13] *= 1.5
    elif emotion_type == 'angry':
        eeg_data[13:] *= 2.0
    elif emotion_type == 'sad':
        eeg_data[:4] *= 2.0
    elif emotion_type == 'calm':
        eeg_data[8:13] *= 2.0
    
    return eeg_data

def generate_synthetic_video_embedding(dim=768):
    return np.random.randn(1, dim).astype(np.float32)

def display_emotion_results(prediction, labels=None):
    emotions = [
        ('Valence', '😊', '😢'),
        ('Arousal', '⚡', '😴'),
        ('Dominance', '💪', '🤝'),
        ('Liking', '❤️', '💔')
    ]
    
    cols = st.columns(4)
    
    for i, (emotion_name, high_emoji, low_emoji) in enumerate(emotions):
        with cols[i]:
            value = prediction[i]
            emoji = high_emoji if value > 5 else low_emoji
            
            st.markdown(f"### {emoji} {emotion_name}")
            st.metric("Predicted", f"{value:.2f}")
            
            if labels is not None:
                st.metric("Ground Truth", f"{labels[i]:.2f}", 
                         delta=f"{value - labels[i]:.2f}")
            
            if value < 3.5:
                level = "Low"
                color = "🔵"
            elif value < 6.5:
                level = "Medium"
                color = "🟡"
            else:
                level = "High"
                color = "🟢"
            
            st.markdown(f"{color} **{level}**")

tab1, tab2, tab3 = st.tabs(["📁 Upload Individual Files", "📊 Upload Fused Dataset", "🧪 Test with Synthetic Data"])

with tab1:
    st.header("Individual File Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("EEG Data")
        eeg_file = st.file_uploader("Upload EEG (.npy)", type=['npy'], key='eeg')
        
        if eeg_file:
            try:
                eeg_data = load_eeg_data(eeg_file)
                st.success(f"EEG loaded: Shape {eeg_data.shape}")
                validate_eeg_shape(eeg_data)
            except Exception as e:
                st.error(f"EEG Error: {str(e)}")
    
    with col2:
        st.subheader("Video Data")
        video_file = st.file_uploader("Upload Video (.mp4, .avi)", type=['mp4', 'avi'], key='video')
        
        if video_file:
            try:
                video_frames = load_video_data(video_file)
                st.success(f"Video loaded: {video_frames.shape[0]} frames")
                
                with st.spinner("Extracting video embeddings..."):
                    video_embedding = extract_video_embeddings(video_frames)
                st.success(f"Embeddings extracted: Shape {video_embedding.shape}")
                validate_video_embedding_shape(video_embedding)
            except Exception as e:
                st.error(f"Video Error: {str(e)}")
    
    if eeg_file and video_file:
        st.divider()
        
        model_type = st.selectbox("Select Model Type", ["Keras (.h5)", "PyTorch (.pt/.pth)"], key='model_tab1')
        
        if st.button("🚀 Run Prediction", key='predict_individual'):
            extensions = [".h5"] if "Keras" in model_type else [".pt", ".pth"]
            
            with st.spinner("Loading models..."):
                models = load_all_fold_models(MODEL_DIR, extensions)
                
                if not models:
                    st.error(f"No models found in {MODEL_DIR} with extensions {extensions}")
                else:
                    st.success(f"Loaded {len(models)} models")
                    
                    st.subheader("🎯 Emotion Prediction Results")
                    predictions = {}
                    
                    for fold_name, model in models.items():
                        if "Keras" in model_type:
                            combined_input = np.concatenate([
                                eeg_data.flatten(),
                                video_embedding.flatten()
                            ]).reshape(1, -1)
                            pred = model.predict(combined_input, verbose=0)
                        else:
                            combined_input = torch.cat([
                                torch.FloatTensor(eeg_data).flatten(),
                                torch.FloatTensor(video_embedding).flatten()
                            ]).unsqueeze(0)
                            with torch.no_grad():
                                pred = model(combined_input).numpy()
                        
                        predictions[fold_name] = pred[0]
                    
                    avg_prediction = np.mean(list(predictions.values()), axis=0)
                    
                    display_emotion_results(avg_prediction)

with tab2:
    st.header("Fused Dataset Upload")
    
    fused_file = st.file_uploader("Upload Fused Dataset (.h5)", type=['h5'], key='fused')
    
    if fused_file:
        col1, col2 = st.columns(2)
        
        with col1:
            subject = st.number_input("Subject ID (1-32)", min_value=1, max_value=32, value=1)
        
        with col2:
            trial = st.number_input("Trial ID (1-40)", min_value=1, max_value=40, value=1)
        
        if st.button("📥 Load Trial Data", key='load_fused'):
            try:
                data = load_fused_data(fused_file, subject, trial)
                
                st.success("✅ Fused data loaded successfully!")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("EEG Shape", f"{data['eeg'].shape}")
                    st.metric("Peripheral Shape", f"{data['peripheral'].shape}")
                
                with col2:
                    st.metric("Video Embedding Shape", f"{data['video_embedding'].shape}")
                
                with col3:
                    st.markdown("**📊 Ground Truth Labels**")
                    labels = ['Valence', 'Arousal', 'Dominance', 'Liking']
                    for i, label in enumerate(labels):
                        st.text(f"{label}: {data['labels'][i]:.2f}")
                
                st.session_state['fused_data'] = data
                
            except Exception as e:
                st.error(f"Error loading fused data: {str(e)}")
        
        if 'fused_data' in st.session_state:
            st.divider()
            
            model_type = st.selectbox("Select Model Type", ["Keras (.h5)", "PyTorch (.pt/.pth)"], key='model_fused')
            
            if st.button("🚀 Run Prediction on Fused Data", key='predict_fused'):
                data = st.session_state['fused_data']
                extensions = [".h5"] if "Keras" in model_type else [".pt", ".pth"]
                
                with st.spinner("Loading models..."):
                    models = load_all_fold_models(MODEL_DIR, extensions)
                    
                    if not models:
                        st.error(f"No models found in {MODEL_DIR}")
                    else:
                        st.success(f"Loaded {len(models)} models")
                        
                        st.subheader("🎯 Emotion Prediction Results")
                        predictions = {}
                        
                        for fold_name, model in models.items():
                            if "Keras" in model_type:
                                combined_input = np.concatenate([
                                    data['eeg'].flatten(),
                                    data['peripheral'].flatten(),
                                    data['video_embedding'].flatten()
                                ]).reshape(1, -1)
                                pred = model.predict(combined_input, verbose=0)
                            else:
                                combined_input = torch.cat([
                                    torch.FloatTensor(data['eeg']).flatten(),
                                    torch.FloatTensor(data['peripheral']).flatten(),
                                    torch.FloatTensor(data['video_embedding']).flatten()
                                ]).unsqueeze(0)
                                with torch.no_grad():
                                    pred = model(combined_input).numpy()
                            
                            predictions[fold_name] = pred[0]
                        
                        avg_prediction = np.mean(list(predictions.values()), axis=0)
                        
                        display_emotion_results(avg_prediction, labels=data['labels'])

with tab3:
    st.header("🧪 Test with Synthetic Data")
    st.info("Generate random synthetic EEG and video data to test your models without uploading files")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Synthetic EEG Configuration")
        emotion_preset = st.selectbox(
            "Select Emotion Preset",
            ["neutral", "happy", "angry", "sad", "calm"],
            help="Different presets adjust EEG frequency bands"
        )
        
        if st.button("🎲 Generate Synthetic EEG", key='gen_eeg'):
            with st.spinner("Generating synthetic EEG data..."):
                synthetic_eeg = generate_synthetic_eeg(emotion_type=emotion_preset)
                st.session_state['synthetic_eeg'] = synthetic_eeg
                st.success(f"✅ Generated EEG: {synthetic_eeg.shape}")
                
                st.markdown("**Frequency Band Characteristics:**")
                if emotion_preset == 'happy':
                    st.write("- Enhanced Alpha (8-13 Hz) in frontal regions")
                    st.write("- Associated with positive valence, moderate arousal")
                elif emotion_preset == 'angry':
                    st.write("- Enhanced Beta/Gamma (13-50 Hz)")
                    st.write("- Associated with high arousal, negative valence")
                elif emotion_preset == 'sad':
                    st.write("- Enhanced Delta (0.5-4 Hz)")
                    st.write("- Associated with low arousal, negative valence")
                elif emotion_preset == 'calm':
                    st.write("- Enhanced Alpha (8-13 Hz)")
                    st.write("- Associated with positive valence, low arousal")
                else:
                    st.write("- Balanced frequency distribution")
    
    with col2:
        st.subheader("Synthetic Video Embedding")
        
        if st.button("🎲 Generate Synthetic Video Embedding", key='gen_video'):
            with st.spinner("Generating synthetic video embedding..."):
                synthetic_video = generate_synthetic_video_embedding()
                st.session_state['synthetic_video'] = synthetic_video
                st.success(f"✅ Generated Video Embedding: {synthetic_video.shape}")
                st.info("Random 768-dimensional feature vector simulating ResNet50 output")
    
    if 'synthetic_eeg' in st.session_state and 'synthetic_video' in st.session_state:
        st.divider()
        
        st.subheader("🔬 Run Prediction on Synthetic Data")
        
        model_type = st.selectbox("Select Model Type", ["Keras (.h5)", "PyTorch (.pt/.pth)"], key='model_synthetic')
        
        if st.button("🚀 Predict Emotion from Synthetic Data", key='predict_synthetic'):
            synthetic_eeg = st.session_state['synthetic_eeg']
            synthetic_video = st.session_state['synthetic_video']
            extensions = [".h5"] if "Keras" in model_type else [".pt", ".pth"]
            
            with st.spinner("Loading models..."):
                models = load_all_fold_models(MODEL_DIR, extensions)
                
                if not models:
                    st.error(f"No models found in {MODEL_DIR} with extensions {extensions}")
                else:
                    st.success(f"Loaded {len(models)} models")
                    
                    st.subheader("🎯 Emotion Prediction Results")
                    predictions = {}
                    
                    for fold_name, model in models.items():
                        if "Keras" in model_type:
                            combined_input = np.concatenate([
                                synthetic_eeg.flatten(),
                                synthetic_video.flatten()
                            ]).reshape(1, -1)
                            pred = model.predict(combined_input, verbose=0)
                        else:
                            combined_input = torch.cat([
                                torch.FloatTensor(synthetic_eeg).flatten(),
                                torch.FloatTensor(synthetic_video).flatten()
                            ]).unsqueeze(0)
                            with torch.no_grad():
                                pred = model(combined_input).numpy()
                        
                        predictions[fold_name] = pred[0]
                    
                    avg_prediction = np.mean(list(predictions.values()), axis=0)
                    
                    display_emotion_results(avg_prediction)
                    
                    st.divider()
                    st.info(f"💡 **Preset used**: {emotion_preset.title()} - Results may vary based on model training")
    
    elif 'synthetic_eeg' in st.session_state or 'synthetic_video' in st.session_state:
        st.warning("⚠️ Generate both EEG and Video data to run predictions")

st.sidebar.header("ℹ️ System Information")
st.sidebar.markdown("""
### Supported Input Formats
- **EEG**: NumPy array (.npy)
  - Shape: (32, 8064)
  - 32 channels, 63 seconds @ 128Hz
  
- **Video**: Video files (.mp4, .avi)
  - Automatically extracts 20 frames
  - Converts to 768-dim embeddings
  
- **Fused Dataset**: HDF5 (.h5)
  - Contains EEG + Peripheral + Video
  - 32 subjects × 40 trials

### Model Types
- Keras Models (.h5)
- PyTorch Models (.pt, .pth)

### DEAP Scale (1-9)
All emotion dimensions are rated on a scale from 1 (low) to 9 (high), following the Self-Assessment Manikin (SAM) rating convention.

### Testing
Use the **Synthetic Data** tab to test your models without uploading real data.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**🔬 Research Note**: This system uses multimodal fusion of EEG signals and facial expressions for emotion recognition based on the DEAP dataset.")

