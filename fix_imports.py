import os

def create_init_files():
    """Create __init__.py files in all directories to make them proper Python packages."""
    directories = [
        "",  # Root directory
        "benchmark",
        "compression",
        "compression/pruning",
        "compression/quantization", 
        "compression/distillation",
        "models",
        "utils",
        "configs",
        "results"
    ]
    
    for directory in directories:
        init_file = os.path.join(directory, "__init__.py")
        if not os.path.exists(init_file):
            with open(init_file, "w") as f:
                pass
            print(f"Created {init_file}")

if __name__ == "__main__":
    create_init_files() 