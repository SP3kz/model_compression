#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simplified implementation of knowledge distillation techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

def distill_knowledge(teacher_model, student_model, trainloader, device, 
                      temperature=4.0, alpha=0.5, epochs=1):
    """
    Perform knowledge distillation from a teacher model to a student model.
    
    Args:
        teacher_model: Trained teacher model
        student_model: Student model to be trained
        trainloader: DataLoader for training data
        device: Device to use for training
        temperature: Temperature for softening probability distributions
        alpha: Weight for balancing soft and hard targets
        epochs: Number of training epochs
        
    Returns:
        Trained student model
    """
    # Make a copy of the student model
    student_model = copy.deepcopy(student_model)
    student_model = student_model.to(device)
    
    # Set teacher model to evaluation mode
    teacher_model.eval()
    teacher_model = teacher_model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    
    # Simplified training loop (just one batch for testing)
    student_model.train()
    
    # Get a single batch for demonstration
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass with teacher model
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        
        # Forward pass with student model
        student_outputs = student_model(inputs)
        
        # Knowledge distillation loss
        # Soft targets from teacher
        soft_targets = F.softmax(teacher_outputs / temperature, dim=1)
        # Log probabilities from student
        log_probs = F.log_softmax(student_outputs / temperature, dim=1)
        # KL divergence loss
        distillation_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
        
        # Standard cross-entropy loss with true labels
        student_loss = criterion(student_outputs, targets)
        
        # Combined loss
        loss = alpha * distillation_loss + (1 - alpha) * student_loss
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Just process one batch for this simplified example
        break
    
    return student_model


def progressive_distillation(teacher_model, student_model, trainloader, device,
                             intermediate_layers=None, epochs=10, learning_rate=0.01):
    """
    Perform progressive knowledge distillation with intermediate layer supervision.
    
    Args:
        teacher_model: Trained teacher model
        student_model: Student model to be trained
        trainloader: DataLoader for training data
        device: Device to use for training
        intermediate_layers: Dictionary mapping student layer names to teacher layer names
        epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        
    Returns:
        Trained student model
    """
    if intermediate_layers is None:
        # Default to standard distillation if no intermediate layers specified
        return distill_knowledge(teacher_model, student_model, trainloader, device, 
                                epochs=epochs, learning_rate=learning_rate)
    
    teacher_model.eval()
    student_model = student_model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    
    # Hooks for intermediate layer outputs
    teacher_features = {}
    student_features = {}
    
    def get_teacher_hook(name):
        def hook(model, input, output):
            teacher_features[name] = output
        return hook
    
    def get_student_hook(name):
        def hook(model, input, output):
            student_features[name] = output
        return hook
    
    # Register hooks
    teacher_handles = []
    student_handles = []
    
    for student_layer, teacher_layer in intermediate_layers.items():
        # This is a simplified approach - in practice, you'd need to get the actual modules
        teacher_module = teacher_model._modules.get(teacher_layer)
        student_module = student_model._modules.get(student_layer)
        
        if teacher_module and student_module:
            teacher_handles.append(teacher_module.register_forward_hook(get_teacher_hook(teacher_layer)))
            student_handles.append(student_module.register_forward_hook(get_student_hook(student_layer)))
    
    # Training loop
    for epoch in range(epochs):
        running_loss = 0.0
        student_model.train()
        
        for inputs, targets in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass with teacher model
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
            
            # Forward pass with student model
            student_outputs = student_model(inputs)
            
            # Standard cross-entropy loss
            ce_loss = criterion(student_outputs, targets)
            
            # Feature matching loss for intermediate layers
            feature_loss = 0.0
            for student_layer, teacher_layer in intermediate_layers.items():
                if student_layer in student_features and teacher_layer in teacher_features:
                    # Adapt feature dimensions if needed
                    s_feat = student_features[student_layer]
                    t_feat = teacher_features[teacher_layer]
                    
                    # Simple L2 loss between features
                    # In practice, you might need to adapt features to match dimensions
                    if s_feat.shape == t_feat.shape:
                        feature_loss += F.mse_loss(s_feat, t_feat)
            
            # Combined loss
            loss = ce_loss + 0.1 * feature_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print statistics
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(trainloader):.4f}")
    
    # Remove hooks
    for handle in teacher_handles + student_handles:
        handle.remove()
    
    return student_model 