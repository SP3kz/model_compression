#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Benchmark script for comparing different model compression techniques.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path to import from other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compression.pruning import prune_model
from compression.quantization import quantize_model
from compression.distillation import distill_knowledge
from models import get_model
from utils.metrics import calculate_metrics
from utils.visualization import plot_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Model Compression Benchmark')
    parser.add_argument('--model', type=str, default='resnet18', 
                        choices=['resnet18', 'mobilenet', 'efficientnet'],
                        help='Model architecture to use')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Dataset to use for benchmarking')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training and evaluation')
    parser.add_argument('--techniques', nargs='+', 
                        default=['pruning', 'quantization', 'distillation'],
                        help='Compression techniques to benchmark')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    return parser.parse_args()


def load_dataset(dataset_name, batch_size):
    """Load the specified dataset."""
    if dataset_name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform_test)
        num_classes = 10
    elif dataset_name == 'cifar100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])
        trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                                 download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                                download=True, transform=transform_test)
        num_classes = 100
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet")
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    
    return trainloader, testloader, num_classes


def benchmark_compression(args):
    """Run benchmarks for different compression techniques."""
    # Set device
    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    trainloader, testloader, num_classes = load_dataset(args.dataset, args.batch_size)
    
    # Get base model
    base_model = get_model(args.model, num_classes)
    base_model = base_model.to(device)
    
    # Train base model if needed
    # For simplicity, we assume the model is pre-trained
    
    # Benchmark each compression technique
    results = {}
    
    # Baseline metrics
    print("Evaluating baseline model...")
    baseline_metrics = calculate_metrics(base_model, testloader, device)
    results['baseline'] = baseline_metrics
    
    # Benchmark each technique
    for technique in args.techniques:
        print(f"Benchmarking {technique}...")
        
        if technique == 'pruning':
            compressed_model = prune_model(base_model.clone(), ratio=0.5)
        elif technique == 'quantization':
            compressed_model = quantize_model(base_model.clone())
        elif technique == 'distillation':
            student_model = get_model(f"small_{args.model}", num_classes)
            compressed_model = distill_knowledge(base_model, student_model, trainloader, device)
        else:
            print(f"Unknown technique: {technique}")
            continue
        
        # Evaluate compressed model
        metrics = calculate_metrics(compressed_model, testloader, device)
        results[technique] = metrics
    
    # Save and visualize results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.model}_{args.dataset}_results.json")
    
    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot comparison
    plot_results(results, os.path.join(args.output_dir, f"{args.model}_{args.dataset}_comparison.png"))
    
    print(f"Results saved to {output_file}")
    return results


if __name__ == "__main__":
    args = parse_args()
    benchmark_compression(args) 