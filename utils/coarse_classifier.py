"""
Stage 1: Coarse-level anatomical region classifier
Classifies images into anatomical regions (brain, abdomen, chest)
"""

import torch
import torch.nn as nn
from .base_model import Base3DCNN, Enhanced3DCNN


class CoarseAnatomicalClassifier(nn.Module):
    """
    Stage 1 classifier for anatomical region localization.
    
    This classifier determines which anatomical region (brain, abdomen, chest)
    a medical image belongs to, mimicking the first step of radiological analysis.
    
    Args:
        architecture (str): Model architecture to use ('base' or 'enhanced')
        num_regions (int): Number of anatomical regions (default: 3)
        dropout_rate (float): Dropout rate (default: 0.3)
    """
    
    def __init__(self, architecture='base', num_regions=3, dropout_rate=0.3, region_names=None):
        super(CoarseAnatomicalClassifier, self).__init__()
        
        self.num_regions = num_regions
        self.architecture = architecture
        
        if architecture == 'enhanced':
            self.model = Enhanced3DCNN(
                in_channels=1,
                num_classes=num_regions,
                dropout_rate=dropout_rate
            )
        else:
            self.model = Base3DCNN(
                in_channels=1,
                num_classes=num_regions,
                dropout_rate=dropout_rate
            )
        
        # Region mapping - use provided mapping or default
        if region_names is None:
            self.region_names = {
                0: 'brain',
                1: 'abdomen', 
                2: 'chest'
            }
        else:
            self.region_names = region_names
    
    def forward(self, x):
        """
        Forward pass to classify anatomical region.
        
        Args:
            x (torch.Tensor): Input 3D medical images
        
        Returns:
            torch.Tensor: Logits for each anatomical region
        """
        return self.model(x)
    
    def predict_region(self, x):
        """
        Predict anatomical region with confidence scores.
        
        Args:
            x (torch.Tensor): Input 3D medical images
        
        Returns:
            tuple: (region_indices, region_names, confidence_scores)
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence_scores, region_indices = torch.max(probabilities, dim=1)
        
        region_names = [self.region_names[idx.item()] for idx in region_indices]
        
        return region_indices, region_names, confidence_scores
    
    def get_region_name(self, region_idx):
        """Get region name from index."""
        return self.region_names.get(region_idx, 'unknown')


class MultiScaleCoarseClassifier(nn.Module):
    """
    Multi-scale coarse classifier that processes images at different resolutions
    for improved anatomical localization.
    """
    
    def __init__(self, num_regions=3, dropout_rate=0.3, region_names=None):
        super(MultiScaleCoarseClassifier, self).__init__()
        
        self.num_regions = num_regions
        
        # Full resolution path
        self.full_res_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2)
        )
        
        # Downsampled path (for global context)
        self.downsample_conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True)
        )
        
        # Combined processing
        self.combined_conv = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_regions)
        )
        
        # Region mapping - use provided mapping or default
        if region_names is None:
            self.region_names = {0: 'brain', 1: 'abdomen', 2: 'chest'}
        else:
            self.region_names = region_names
    
    def forward(self, x):
        # Full resolution path
        full_res = self.full_res_conv(x)
        
        # Downsampled path
        downsample = self.downsample_conv(x)
        
        # Concatenate features
        combined = torch.cat([full_res, downsample], dim=1)
        
        # Process combined features
        features = self.combined_conv(combined)
        features = features.view(features.size(0), -1)
        
        # Classification
        logits = self.fc(features)
        return logits
