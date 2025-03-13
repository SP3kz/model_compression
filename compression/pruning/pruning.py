#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified implementation of model pruning techniques.
"""

import torch
import torch.nn as nn
import copy

def prune_model(model, ratio=0.5, method='l1_unstructured'):
    """
    Prune a model by removing weights with lowest magnitudes.
    
    Args:
        model: PyTorch model to prune
        ratio: Percentage of weights to prune (0.0 to 1.0)
        method: Pruning method to use
        
    Returns:
        Pruned model
    """
    model_copy = copy.deepcopy(model)
    
    # Apply pruning to all convolutional and linear layers
    for name, module in model_copy.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # Get the absolute values of weights
            abs_weights = torch.abs(module.weight.data)
            # Calculate threshold for pruning
            threshold = torch.quantile(abs_weights.flatten(), ratio)
            # Create a mask for weights to keep
            mask = abs_weights > threshold
            # Apply the mask
            module.weight.data[~mask] = 0
    
    return model_copy


def iterative_pruning(model, trainloader, testloader, device, 
                      initial_ratio=0.2, final_ratio=0.8, steps=4, 
                      epochs_per_step=3, learning_rate=0.001):
    """
    Perform iterative pruning with retraining between pruning steps.
    
    Args:
        model: PyTorch model to prune
        trainloader: DataLoader for training data
        testloader: DataLoader for testing data
        device: Device to use for training
        initial_ratio: Initial pruning ratio
        final_ratio: Final pruning ratio
        steps: Number of pruning steps
        epochs_per_step: Number of retraining epochs per step
        learning_rate: Learning rate for retraining
        
    Returns:
        Pruned model
    """
    model_copy = copy.deepcopy(model)
    model_copy = model_copy.to(device)
    
    # Calculate pruning schedule
    pruning_ratios = [initial_ratio + (final_ratio - initial_ratio) * i / (steps - 1) 
                      for i in range(steps)]
    
    criterion = nn.CrossEntropyLoss()
    
    for i, ratio in enumerate(pruning_ratios):
        print(f"Pruning step {i+1}/{steps}, ratio: {ratio:.2f}")
        
        # Prune the model
        model_copy = prune_model(model_copy, ratio=ratio if i == 0 else 
                                (ratio - pruning_ratios[i-1]) / (1 - pruning_ratios[i-1]))
        
        # Retrain the model
        optimizer = torch.optim.Adam(model_copy.parameters(), lr=learning_rate)
        
        for epoch in range(epochs_per_step):
            # Training
            model_copy.train()
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model_copy(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
            
            # Evaluation
            model_copy.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model_copy(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            print(f"Epoch {epoch+1}/{epochs_per_step}, Loss: {running_loss/len(trainloader):.4f}, "
                  f"Accuracy: {100 * correct / total:.2f}%")
    
    return model_copy 