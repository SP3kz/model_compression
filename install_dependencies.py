import subprocess
import sys
import os

def install_dependencies():
    """Install all required dependencies for the model compression toolkit."""
    print("Installing dependencies...")
    
    # The correct packages for PyTorch
    dependencies = [
        "torch",
        "torchvision",
        "numpy",
        "matplotlib",
        "pandas",
        "seaborn",
        "tqdm",
        "scikit-learn"
    ]
    
    for package in dependencies:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    print("All dependencies installed successfully!")

if __name__ == "__main__":
    install_dependencies() 