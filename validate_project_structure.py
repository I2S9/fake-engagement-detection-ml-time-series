#!/usr/bin/env python3
"""
Validation script for project structure completeness.
Checks that the project contains all 5 essential blocks:
1. Realistic simulator
2. Analytical exploration
3. Automatic annotation (true/false anomalies)
4. Detailed visualization
5. Clear structure
"""
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def check_1_realistic_simulator() -> dict:
    """
    Check 1: Realistic simulator exists.
    
    Checks:
    - Simulation module exists
    - Can generate dataset
    - Multiple profiles and attack types
    """
    print("\n" + "="*60)
    print("CHECK 1: Realistic Simulator")
    print("="*60)
    
    results = {
        'passed': True,
        'simulator_module': False,
        'profiles_found': [],
        'attack_types_found': [],
        'can_generate': False,
        'issues': []
    }
    
    # Check simulator module
    simulator_path = project_root / "src" / "data" / "simulate_behaviors.py"
    if simulator_path.exists():
        results['simulator_module'] = True
        print("  ✓ Simulator module found: src/data/simulate_behaviors.py")
        
        # Check for profiles and attack types in code
        with open(simulator_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract profiles
            if 'profiles = [' in content:
                import re
                profiles_match = re.search(r'profiles = \[(.*?)\]', content, re.DOTALL)
                if profiles_match:
                    profiles_str = profiles_match.group(1)
                    profiles = [p.strip().strip('"\'') for p in profiles_str.split(',')]
                    results['profiles_found'] = [p for p in profiles if p]
                    print(f"  ✓ Profiles found: {len(results['profiles_found'])}")
                    print(f"    {', '.join(results['profiles_found'][:5])}...")
            
            # Extract attack types
            if 'attack_types = [' in content:
                import re
                attacks_match = re.search(r'attack_types = \[(.*?)\]', content, re.DOTALL)
                if attacks_match:
                    attacks_str = attacks_match.group(1)
                    attacks = [a.strip().strip('"\'') for a in attacks_str.split(',')]
                    results['attack_types_found'] = [a for a in attacks if a]
                    print(f"  ✓ Attack types found: {len(results['attack_types_found'])}")
                    print(f"    {', '.join(results['attack_types_found'][:5])}...")
        
        # Check if can generate dataset
        make_dataset_path = project_root / "src" / "data" / "make_dataset.py"
        if make_dataset_path.exists():
            results['can_generate'] = True
            print("  ✓ Dataset generation script found: src/data/make_dataset.py")
        else:
            results['issues'].append("Dataset generation script not found")
            results['passed'] = False
    else:
        results['issues'].append("Simulator module not found")
        results['passed'] = False
        print("  ✗ Simulator module not found")
    
    if results['passed'] and len(results['profiles_found']) >= 5 and len(results['attack_types_found']) >= 5:
        print("\n  ✓ Realistic simulator validated")
    else:
        print("\n  ⚠ Simulator may be incomplete")
        if len(results['profiles_found']) < 5:
            results['issues'].append(f"Only {len(results['profiles_found'])} profiles found (need at least 5)")
        if len(results['attack_types_found']) < 5:
            results['issues'].append(f"Only {len(results['attack_types_found'])} attack types found (need at least 5)")
    
    return results


def check_2_analytical_exploration() -> dict:
    """
    Check 2: Analytical exploration exists.
    
    Checks:
    - Exploration notebook exists
    - Contains statistical analysis
    - Contains pattern analysis
    """
    print("\n" + "="*60)
    print("CHECK 2: Analytical Exploration")
    print("="*60)
    
    results = {
        'passed': True,
        'exploration_notebook': False,
        'analysis_types': [],
        'issues': []
    }
    
    # Check exploration notebook
    exploration_nb = project_root / "notebooks" / "01_exploration.ipynb"
    if exploration_nb.exists():
        results['exploration_notebook'] = True
        print("  ✓ Exploration notebook found: notebooks/01_exploration.ipynb")
        
        # Check notebook content
        import json
        with open(exploration_nb, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        all_code = ' '.join([
            ''.join(cell.get('source', [])) 
            for cell in nb['cells'] 
            if cell.get('cell_type') == 'code'
        ]).lower()
        
        # Check for analysis types
        if 'histogram' in all_code or 'hist(' in all_code:
            results['analysis_types'].append('histograms')
            print("  ✓ Histograms analysis found")
        
        if 'correlation' in all_code or 'corr(' in all_code:
            results['analysis_types'].append('correlation')
            print("  ✓ Correlation analysis found")
        
        if 'statistic' in all_code or 'describe(' in all_code:
            results['analysis_types'].append('statistics')
            print("  ✓ Statistical analysis found")
        
        if 'pattern' in all_code or 'temporal' in all_code:
            results['analysis_types'].append('patterns')
            print("  ✓ Pattern analysis found")
        
        if 'distribution' in all_code:
            results['analysis_types'].append('distributions')
            print("  ✓ Distribution analysis found")
        
        if len(results['analysis_types']) >= 3:
            print(f"\n  ✓ Analytical exploration validated ({len(results['analysis_types'])} analysis types)")
        else:
            results['issues'].append(f"Only {len(results['analysis_types'])} analysis types found")
            results['passed'] = False
    else:
        results['issues'].append("Exploration notebook not found")
        results['passed'] = False
        print("  ✗ Exploration notebook not found")
    
    return results


def check_3_automatic_annotation() -> dict:
    """
    Check 3: Automatic annotation (true/false anomalies).
    
    Checks:
    - Labels exist in dataset
    - Anomaly windows are marked
    - Labels are used in visualizations
    """
    print("\n" + "="*60)
    print("CHECK 3: Automatic Annotation")
    print("="*60)
    
    results = {
        'passed': True,
        'labels_in_dataset': False,
        'anomaly_windows': False,
        'labels_in_plots': False,
        'issues': []
    }
    
    # Check dataset structure
    data_path = project_root / "data" / "raw" / "engagement.parquet"
    if data_path.exists():
        try:
            import pandas as pd
            df = pd.read_parquet(data_path)
            
            # Check for labels
            if 'is_fake_series' in df.columns or 'label' in df.columns:
                results['labels_in_dataset'] = True
                print("  ✓ Labels found in dataset (is_fake_series or label)")
            else:
                results['issues'].append("Labels not found in dataset")
                results['passed'] = False
            
            # Check for anomaly windows
            if 'is_anomaly_window' in df.columns:
                results['anomaly_windows'] = True
                anomaly_count = df['is_anomaly_window'].sum()
                print(f"  ✓ Anomaly windows found: {anomaly_count} windows")
            else:
                results['issues'].append("Anomaly windows not found in dataset")
                results['passed'] = False
        except Exception as e:
            results['issues'].append(f"Error reading dataset: {e}")
            results['passed'] = False
    else:
        print("  ⚠ Dataset not found (may need to generate it)")
    
    # Check if labels are used in plots
    exploration_nb = project_root / "notebooks" / "01_exploration.ipynb"
    if exploration_nb.exists():
        import json
        with open(exploration_nb, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        all_code = ' '.join([
            ''.join(cell.get('source', [])) 
            for cell in nb['cells'] 
            if cell.get('cell_type') == 'code'
        ]).lower()
        
        if 'label' in all_code or 'is_fake' in all_code or 'anomaly' in all_code:
            results['labels_in_plots'] = True
            print("  ✓ Labels used in visualizations")
        else:
            results['issues'].append("Labels not used in visualizations")
    
    if results['labels_in_dataset'] and results['anomaly_windows']:
        print("\n  ✓ Automatic annotation validated")
    else:
        print("\n  ⚠ Automatic annotation may be incomplete")
        results['passed'] = False
    
    return results


def check_4_detailed_visualization() -> dict:
    """
    Check 4: Detailed visualization.
    
    Checks:
    - Multiple visualization types
    - Plots are saved
    - Visualization module exists
    """
    print("\n" + "="*60)
    print("CHECK 4: Detailed Visualization")
    print("="*60)
    
    results = {
        'passed': True,
        'visualization_module': False,
        'plot_types': [],
        'saved_plots': 0,
        'issues': []
    }
    
    # Check visualization module
    viz_module = project_root / "src" / "visualization" / "plots.py"
    if viz_module.exists():
        results['visualization_module'] = True
        print("  ✓ Visualization module found: src/visualization/plots.py")
        
        # Check for plot functions
        with open(viz_module, 'r', encoding='utf-8') as f:
            content = f.read()
            if 'def plot_' in content:
                import re
                plot_functions = re.findall(r'def (plot_\w+)', content)
                results['plot_types'] = plot_functions
                print(f"  ✓ Plot functions found: {len(plot_functions)}")
                print(f"    {', '.join(plot_functions[:5])}...")
    else:
        results['issues'].append("Visualization module not found")
        results['passed'] = False
        print("  ✗ Visualization module not found")
    
    # Check saved plots
    plots_dir = project_root / "outputs" / "figures"
    if plots_dir.exists():
        plot_files = list(plots_dir.glob("*.png"))
        results['saved_plots'] = len(plot_files)
        print(f"  ✓ Saved plots found: {len(plot_files)} PNG files")
        if len(plot_files) >= 10:
            print("  ✓ Strong visualization coverage")
        else:
            print(f"  ⚠ Only {len(plot_files)} plots found (may need more)")
    else:
        print("  ⚠ Plots directory not found")
    
    # Check notebooks for visualization
    notebooks_dir = project_root / "notebooks"
    if notebooks_dir.exists():
        notebooks = list(notebooks_dir.glob("*.ipynb"))
        viz_notebooks = 0
        for nb_path in notebooks:
            import json
            try:
                with open(nb_path, 'r', encoding='utf-8') as f:
                    nb = json.load(f)
                all_code = ' '.join([
                    ''.join(cell.get('source', [])) 
                    for cell in nb['cells'] 
                    if cell.get('cell_type') == 'code'
                ]).lower()
                if 'plt.' in all_code or 'plot(' in all_code or 'savefig' in all_code:
                    viz_notebooks += 1
            except:
                pass
        print(f"  ✓ Notebooks with visualizations: {viz_notebooks}/{len(notebooks)}")
    
    if results['visualization_module'] and results['saved_plots'] >= 5:
        print("\n  ✓ Detailed visualization validated")
    else:
        print("\n  ⚠ Visualization may be incomplete")
        results['passed'] = False
    
    return results


def check_5_clear_structure() -> dict:
    """
    Check 5: Clear structure.
    
    Checks:
    - Required directories exist
    - Organized structure
    - README exists
    """
    print("\n" + "="*60)
    print("CHECK 5: Clear Structure")
    print("="*60)
    
    results = {
        'passed': True,
        'directories_found': [],
        'directories_missing': [],
        'readme_exists': False,
        'issues': []
    }
    
    required_dirs = {
        'data': project_root / "data",
        'notebooks': project_root / "notebooks",
        'src': project_root / "src",
        'outputs': project_root / "outputs",
        'config': project_root / "config",
    }
    
    print("\n  Checking directory structure:")
    for name, path in required_dirs.items():
        if path.exists():
            results['directories_found'].append(name)
            print(f"    ✓ {name}/")
        else:
            results['directories_missing'].append(name)
            print(f"    ✗ {name}/ (missing)")
            results['issues'].append(f"Directory {name}/ not found")
            results['passed'] = False
    
    # Check subdirectories
    if (project_root / "data").exists():
        subdirs = ['raw', 'processed']
        for subdir in subdirs:
            subdir_path = project_root / "data" / subdir
            if subdir_path.exists():
                print(f"      ✓ data/{subdir}/")
            else:
                print(f"      ⚠ data/{subdir}/ (optional)")
    
    if (project_root / "src").exists():
        subdirs = ['data', 'features', 'models', 'training', 'visualization']
        for subdir in subdirs:
            subdir_path = project_root / "src" / subdir
            if subdir_path.exists():
                print(f"      ✓ src/{subdir}/")
            else:
                print(f"      ⚠ src/{subdir}/ (optional)")
    
    # Check README
    readme_path = project_root / "README.md"
    if readme_path.exists():
        results['readme_exists'] = True
        print("\n  ✓ README.md found")
    else:
        results['issues'].append("README.md not found")
        results['passed'] = False
        print("\n  ✗ README.md not found")
    
    if len(results['directories_found']) >= 4 and results['readme_exists']:
        print("\n  ✓ Clear structure validated")
    else:
        print("\n  ⚠ Structure may be incomplete")
        results['passed'] = False
    
    return results


def main():
    """Main validation function."""
    print("="*60)
    print("PROJECT STRUCTURE COMPLETENESS VALIDATION")
    print("="*60)
    
    # Run all checks
    check_results = {}
    
    check_results['check_1'] = check_1_realistic_simulator()
    check_results['check_2'] = check_2_analytical_exploration()
    check_results['check_3'] = check_3_automatic_annotation()
    check_results['check_4'] = check_4_detailed_visualization()
    check_results['check_5'] = check_5_clear_structure()
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = all(
        check_results[f'check_{i}']['passed'] 
        for i in range(1, 6)
    )
    
    print("\nEssential Blocks:")
    print(f"  1. Realistic Simulator: {'✓' if check_results['check_1']['passed'] else '✗'}")
    print(f"  2. Analytical Exploration: {'✓' if check_results['check_2']['passed'] else '✗'}")
    print(f"  3. Automatic Annotation: {'✓' if check_results['check_3']['passed'] else '✗'}")
    print(f"  4. Detailed Visualization: {'✓' if check_results['check_4']['passed'] else '✗'}")
    print(f"  5. Clear Structure: {'✓' if check_results['check_5']['passed'] else '✗'}")
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL ESSENTIAL BLOCKS PRESENT")
        print("="*60)
        print("\nThe project contains all 5 essential blocks:")
        print("  ✓ Realistic simulator with multiple profiles and attack types")
        print("  ✓ Analytical exploration with comprehensive analysis")
        print("  ✓ Automatic annotation (labels and anomaly windows)")
        print("  ✓ Detailed visualization with strong coverage")
        print("  ✓ Clear structure with organized directories")
        print("\nThe project is complete and professional!")
    else:
        print("\n" + "="*60)
        print("⚠ SOME ESSENTIAL BLOCKS MAY BE MISSING")
        print("="*60)
        print("\nPlease review the check results above.")


if __name__ == "__main__":
    main()

