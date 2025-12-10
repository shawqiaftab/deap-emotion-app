import torch
import torch.nn as nn
import tensorflow as tf

class MultimodalFusionModel(nn.Module):
    def __init__(self, eeg_input_dim=320, video_input_dim=768, hidden_dim=256, num_classes=2, dropout=0.4):
        super(MultimodalFusionModel, self).__init__()
        
        self.eeg_branch = nn.Sequential(
            nn.Linear(eeg_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.video_branch = nn.Sequential(
            nn.Linear(video_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, eeg, video):
        eeg_out = self.eeg_branch(eeg)
        video_out = self.video_branch(video)
        fused = torch.cat([eeg_out, video_out], dim=1)
        output = self.fusion(fused)
        return output

def load_pytorch_model(model_path):
    model = MultimodalFusionModel()
    state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_keras_model(model_path):
    return tf.keras.models.load_model(model_path)
