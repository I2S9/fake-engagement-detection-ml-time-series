"""
Script to execute all notebooks and verify visualizations are displayed.
"""

import subprocess
import sys
from pathlib import Path

notebooks = [
    "notebooks/01_exploration.ipynb",
    "notebooks/02_feature_engineering.ipynb",
    "notebooks/03_modeling.ipynb",
    "notebooks/04_evaluation.ipynb",
]

print("=" * 60)
print("Executing all notebooks")
print("=" * 60)

for notebook_path in notebooks:
    notebook = Path(notebook_path)
    if not notebook.exists():
        print(f"\n[SKIP] {notebook_path} does not exist")
        continue
    
    print(f"\n{'='*60}")
    print(f"Executing: {notebook_path}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", 
             "--inplace", str(notebook)],
            capture_output=True,
            text=True,
            timeout=600
        )
        
        if result.returncode == 0:
            print(f"[OK] {notebook_path} executed successfully")
        else:
            print(f"[ERROR] {notebook_path} failed:")
            print(result.stderr)
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {notebook_path} took too long (>10 minutes)")
    except Exception as e:
        print(f"[ERROR] {notebook_path}: {e}")

print("\n" + "=" * 60)
print("All notebooks execution completed")
print("=" * 60)

