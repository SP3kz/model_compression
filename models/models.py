#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified model definitions and utilities.
"""

import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc = nn.Linear(32 * 8 * 8, num_classes)
    
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = self.fc(x)
        return x

def get_model(model_name, num_classes=10, pretrained=False):
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        
    Returns:
        PyTorch model
    """
    if model_name == 'simple':
        return SimpleModel(num_classes=num_classes)
    elif model_name == 'small_simple':
        # A smaller version for distillation
        model = SimpleModel(num_classes=num_classes)
        # Reduce number of filters
        model.conv1 = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        model.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        model.fc = nn.Linear(16 * 8 * 8, num_classes)
        return model
    else:
        # Default to simple model
        print(f"Model {model_name} not found, using SimpleModel instead")
        return SimpleModel(num_classes=num_classes)


class CustomCNN(nn.Module):
    """
    A simple custom CNN for experimentation.
    """
    def __init__(self, num_classes=10):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x 