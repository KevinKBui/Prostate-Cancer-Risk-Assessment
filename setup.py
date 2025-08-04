#!/usr/bin/env python3
"""
Setup script for Prostate Cancer Risk Prediction MLOps Pipeline
This script helps with initial project setup and validation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import pandas as pd
import numpy as np


def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def run_command(command, description="", check=True):
    """Run a shell command with error handling."""
    if description:
        print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(command, shell=True, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False


def check_prerequisites():
    """Check if required tools are installed."""
    print_section("Checking Prerequisites")
    
    tools = {
        "python": "python --version",
        "pip": "pip --version",
        "docker": "docker --version",
        "docker-compose": "docker-compose --version",
        "git": "git --version"
    }
    
    missing = []
    for tool, command in tools.items():
        if run_command(command, f"Checking {tool}", check=False):
            print(f"✅ {tool} is installed")
        else:
            print(f"❌ {tool} is not installed")
            missing.append(tool)
    
    if missing:
        print(f"\n⚠️  Missing tools: {', '.join(missing)}")
        print("Please install missing tools before continuing.")
        return False
    
    print("\n✅ All prerequisites are installed!")
    return True


def create_directories():
    """Create necessary project directories."""
    print_section("Creating Project Directories")
    
    directories = [
        "models/run_id",
        "models/artifacts", 
        "logs",
        "monitoring/reports",
        "monitoring/dashboards",
        "Data/processed",
        "artifacts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def create_sample_data():
    """Create sample dataset for testing."""
    print_section("Creating Sample Dataset")
    
    data_path = Path("Data/raw/synthetic_prostate_cancer_risk.csv")
    
    if data_path.exists():
        print(f"✅ Dataset already exists at {data_path}")
        return
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'id': range(1, n_samples + 1),
        'age': np.random.randint(18, 80, n_samples),
        'bmi': np.random.uniform(15, 40, n_samples),
        'sleep_hours': np.random.uniform(4, 12, n_samples),
        'family_history': np.random.choice(['Yes', 'No'], n_samples),
        'diet': np.random.choice(['Healthy', 'Moderate', 'Unhealthy'], n_samples),
        'exercise_frequency': np.random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely'], n_samples),
        'smoking_status': np.random.choice(['Never', 'Former', 'Current'], n_samples),
        'alcohol_consumption': np.random.choice(['None', 'Light', 'Moderate', 'Heavy'], n_samples),
        'risk_level': np.random.randint(0, 3, n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(data_path, index=False)
    print(f"✅ Created sample dataset with {n_samples} samples at {data_path}")


def install_dependencies():
    """Install Python dependencies."""
    print_section("Installing Dependencies")
    
    if not run_command("pip install -r requirements.txt", "Installing Python packages"):
        return False
    
    print("✅ Dependencies installed successfully!")
    return True


def setup_pre_commit():
    """Setup pre-commit hooks."""
    print_section("Setting up Pre-commit Hooks")
    
    if not run_command("pip install pre-commit", "Installing pre-commit"):
        return False
    
    if not run_command("pre-commit install", "Installing pre-commit hooks"):
        return False
    
    print("✅ Pre-commit hooks installed successfully!")
    return True


def validate_setup():
    """Validate the setup by running basic tests."""
    print_section("Validating Setup")
    
    # Test imports
    try:
        import pandas as pd
        import numpy as np
        import mlflow
        import xgboost as xgb
        import flask
        import prefect
        import evidently
        print("✅ All required packages can be imported")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test data loading
    try:
        data_path = "Data/raw/synthetic_prostate_cancer_risk.csv"
        df = pd.read_csv(data_path)
        print(f"✅ Sample data loaded successfully ({len(df)} rows)")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Test model training (quick version)
    try:
        from src.orchestration import preprocess_data, load_data
        df = load_data("Data/raw/synthetic_prostate_cancer_risk.csv")
        X, y = preprocess_data(df)
        print("✅ Data preprocessing works correctly")
    except Exception as e:
        print(f"❌ Error in preprocessing: {e}")
        return False
    
    print("\n✅ Setup validation completed successfully!")
    return True


def print_next_steps():
    """Print next steps for the user."""
    print_section("Next Steps")
    
    steps = [
        "1. Start MLflow server: make mlflow-server",
        "2. Train the model: make train",
        "3. Start web service: make serve",
        "4. Or deploy all services: make prod-deploy",
        "5. Run tests: make test",
        "6. Check monitoring: make monitor",
        "",
        "Web Service: http://localhost:9696",
        "MLflow UI: http://localhost:5000",
        "Grafana: http://localhost:3000",
        "",
        "For more commands: make help"
    ]
    
    for step in steps:
        print(step)


def main():
    """Main setup function."""
    print("🚀 Prostate Cancer Risk Prediction - MLOps Pipeline Setup")
    print("This script will set up your development environment.")
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create sample data
    create_sample_data()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Setup pre-commit hooks
    setup_pre_commit()
    
    # Validate setup
    if not validate_setup():
        print("❌ Setup validation failed")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()
    
    print("\n🎉 Setup completed successfully!")
    print("Your MLOps pipeline is ready to use!")


if __name__ == "__main__":
    main()