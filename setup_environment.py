"""Environment setup script for Urdu Sentiment Analysis project."""

import torch
import sys
import subprocess
from pathlib import Path

def check_gpu_availability():
    """Check if GPU is available and display information."""
    print("=== GPU Availability Check ===")
    
    if torch.cuda.is_available():
        print(f"✅ CUDA is available!")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Test GPU functionality
        try:
            test_tensor = torch.randn(100, 100).cuda()
            result = torch.mm(test_tensor, test_tensor.t())
            print("✅ GPU computation test passed!")
        except Exception as e:
            print(f"❌ GPU computation test failed: {e}")
    else:
        print("❌ CUDA is not available. Using CPU.")
    
    print()

def check_dependencies():
    """Check if all required dependencies are installed."""
    print("=== Dependency Check ===")
    
    required_packages = [
        "torch", "transformers", "fastapi", "uvicorn", 
        "pydantic", "numpy", "pandas", "scikit-learn",
        "datasets", "accelerate", "pytest"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
    else:
        print("\n✅ All dependencies are installed!")
    
    print()

def verify_project_structure():
    """Verify that all necessary directories exist."""
    print("=== Project Structure Check ===")
    
    required_dirs = [
        "src/models", "src/data", "src/api", "src/evaluation",
        "tests", "config", "data", "models", "logs", "cache"
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✅ {dir_path}/")
        else:
            print(f"❌ {dir_path}/ - Missing")
            path.mkdir(parents=True, exist_ok=True)
            print(f"   Created {dir_path}/")
    
    print()

def main():
    """Main setup function."""
    print("🚀 Urdu Sentiment Analysis - Environment Setup")
    print("=" * 50)
    
    check_gpu_availability()
    check_dependencies()
    verify_project_structure()
    
    print("🎉 Environment setup complete!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run tests: pytest tests/")
    print("3. Start development!")

if __name__ == "__main__":
    main()