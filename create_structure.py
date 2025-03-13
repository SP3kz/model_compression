import os

# Create directory structure
directories = [
    "benchmark",
    "compression",
    "compression/pruning",
    "compression/quantization",
    "compression/distillation",
    "models",
    "utils",
    "data",
    "configs",
    "results"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)
    # Create __init__.py in each directory to make them proper Python packages
    with open(os.path.join(directory, "__init__.py"), "w") as f:
        pass

print("Directory structure created successfully!") 