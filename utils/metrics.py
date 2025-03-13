#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified metrics for evaluating model performance and compression.
"""

import time
import torch
import numpy as np

def calculate_metrics(model, dataloader, device):
    """
    Calculate various metrics for a model.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for evaluation
        device: Device to use for evaluation
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    model = model.to(device)
    
    # Accuracy
    correct = 0
    total = 0
    
    # Inference time
    inference_times = []
    
    # Memory usage
    param_count = sum(p.numel() for p in model.parameters())
    model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)  # MB
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Measure inference time
            start_time = time.time()
            outputs = model(inputs)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    # Calculate metrics
    accuracy = 100 * correct / total if total > 0 else 0
    avg_inference_time = np.mean(inference_times) * 1000 if inference_times else 0  # ms
    
    # Sparsity (percentage of zero weights)
    zero_weights = 0
    total_weights = 0
    for param in model.parameters():
        if param.dim() > 1:  # Only consider weight matrices, not biases
            zero_weights += torch.sum(param == 0).item()
            total_weights += param.numel()
    
    sparsity = 100 * zero_weights / total_weights if total_weights > 0 else 0
    
    metrics = {
        'accuracy': accuracy,
        'inference_time_ms': avg_inference_time,
        'model_size_mb': model_size,
        'parameter_count': param_count,
        'sparsity_percent': sparsity
    }
    
    return metrics


def count_ops(model, input_size=(1, 3, 224, 224), device='cpu'):
    """
    Count the number of floating-point operations (FLOPs) in a model.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size
        device: Device to use
        
    Returns:
        Number of FLOPs
    """
    # This is a placeholder - in a real implementation, you would use a library
    # like thop or ptflops to count operations
    
    # Example with ptflops:
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model, input_size[1:], as_strings=False)
    # return macs
    
    return 0  # Placeholder 