"""
Stage 2: Fine-level pathology classifier
Region-specific classifiers for disease detection
"""

import torch
import torch.nn as nn
from .base_model import Base3DCNN, Enhanced3DCNN


class FinePathologyClassifier(nn.Module):
    """
    Stage 2 classifier for pathology identification within anatomical regions.
    
    This creates specialized classifiers for each anatomical region to identify
    specific diseases or abnormalities, following the coarse localization.
    
    Args:
        region_name (str): Anatomical region ('brain', 'abdomen', 'chest')
        num_pathologies (int): Number of pathology classes for this region
        architecture (str): Model architecture ('base' or 'enhanced')
        dropout_rate (float): Dropout rate
    """
    
    def __init__(self, region_name, num_pathologies, architecture='base', dropout_rate=0.3):
        super(FinePathologyClassifier, self).__init__()
        
        self.region_name = region_name
        self.num_pathologies = num_pathologies
        self.architecture = architecture
        
        if architecture == 'enhanced':
            self.model = Enhanced3DCNN(
                in_channels=1,
                num_classes=num_pathologies,
                dropout_rate=dropout_rate
            )
        else:
            self.model = Base3DCNN(
                in_channels=1,
                num_classes=num_pathologies,
                dropout_rate=dropout_rate
            )
    
    def forward(self, x):
        """
        Forward pass for pathology classification.
        
        Args:
            x (torch.Tensor): Input 3D medical images from specific region
        
        Returns:
            torch.Tensor: Logits for each pathology class
        """
        return self.model(x)
    
    def predict_pathology(self, x):
        """
        Predict pathology with confidence scores.
        
        Args:
            x (torch.Tensor): Input images
        
        Returns:
            tuple: (pathology_indices, confidence_scores)
        """
        logits = self.forward(x)
        probabilities = torch.softmax(logits, dim=1)
        confidence_scores, pathology_indices = torch.max(probabilities, dim=1)
        
        return pathology_indices, confidence_scores


class RegionSpecificPathologyNetwork(nn.Module):
    """
    Container for multiple region-specific pathology classifiers.
    
    Manages separate specialized models for different anatomical regions,
    allowing independent training and inference for each region.
    """
    
    def __init__(self, region_configs, architecture='base', dropout_rate=0.3):
        """
        Args:
            region_configs (dict): Dict mapping region names to number of pathologies
                Example: {'brain': 5, 'abdomen': 8, 'chest': 6}
            architecture (str): Architecture type
            dropout_rate (float): Dropout rate
        """
        super(RegionSpecificPathologyNetwork, self).__init__()
        
        self.region_configs = region_configs
        self.classifiers = nn.ModuleDict()
        
        for region_name, num_pathologies in region_configs.items():
            self.classifiers[region_name] = FinePathologyClassifier(
                region_name=region_name,
                num_pathologies=num_pathologies,
                architecture=architecture,
                dropout_rate=dropout_rate
            )
    
    def forward(self, x, region_name):
        """
        Forward pass through region-specific classifier.
        
        Args:
            x (torch.Tensor): Input images
            region_name (str): Name of the anatomical region
        
        Returns:
            torch.Tensor: Pathology logits
        """
        if region_name not in self.classifiers:
            raise ValueError(f"Unknown region: {region_name}")
        
        return self.classifiers[region_name](x)
    
    def get_classifier(self, region_name):
        """Get classifier for specific region."""
        return self.classifiers.get(region_name)


class AttentionFineClassifier(nn.Module):
    """
    Fine classifier with attention mechanism to focus on relevant image regions.
    """
    
    def __init__(self, region_name, num_pathologies, dropout_rate=0.3):
        super(AttentionFineClassifier, self).__init__()
        
        self.region_name = region_name
        self.num_pathologies = num_pathologies
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True)
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_pathologies)
        )
    
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Generate attention map
        attention_map = self.attention(features)
        
        # Apply attention
        attended_features = features * attention_map
        
        # Classify
        logits = self.classifier(attended_features)
        
        return logits
