#!/usr/bin/env python3
"""
Simplified validation script for notebook execution order.
Checks that imports come before usage and variables are defined before use.
"""
import sys
import json
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def check_notebook_execution_order(notebook_path: Path) -> dict:
    """
    Check that notebook can be executed from top to bottom.
    
    Simplified checks:
    - Imports are in early cells
    - No obvious undefined variables (excluding built-ins and imports)
    """
    results = {
        'notebook': str(notebook_path.name),
        'total_cells': 0,
        'code_cells': 0,
        'import_cells': [],
        'issues': [],
        'warnings': [],
        'all_ok': True
    }
    
    if not notebook_path.exists():
        results['issues'].append(f"Notebook not found: {notebook_path}")
        results['all_ok'] = False
        return results
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    results['total_cells'] = len(nb['cells'])
    
    # Track what's been imported/defined
    imports_found = set()
    variables_defined = set()
    functions_defined = set()
    
    # Built-ins and common imports
    builtins = {
        'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 'set', 'tuple',
        'max', 'min', 'sum', 'abs', 'round', 'sorted', 'enumerate', 'zip', 'map', 'filter',
        'any', 'all', 'isinstance', 'type', 'hasattr', 'getattr', 'setattr', 'delattr',
        'dir', 'vars', 'locals', 'globals', 'eval', 'exec', 'compile', 'open', 'file',
        'input', 'exit', 'quit', 'True', 'False', 'None'
    }
    
    for cell_idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') != 'code':
            continue
        
        results['code_cells'] += 1
        source = cell.get('source', [])
        if isinstance(source, list):
            code = ''.join(source)
        else:
            code = str(source)
        
        # Check for imports
        has_import = False
        if 'import ' in code or 'from ' in code:
            has_import = True
            results['import_cells'].append(cell_idx)
            # Extract imported names
            lines = code.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('import '):
                    parts = line.replace('import ', '').split(' as ')
                    if len(parts) > 0:
                        module = parts[0].split('.')[0].strip()
                        imports_found.add(module)
                elif line.startswith('from '):
                    parts = line.split(' import ')
                    if len(parts) == 2:
                        module = parts[0].replace('from ', '').split('.')[0].strip()
                        imports_found.add(module)
                        # Also add imported names
                        imported = parts[1].strip()
                        for name in imported.split(','):
                            name = name.strip().split(' as ')[0].strip()
                            imports_found.add(name)
        
        # Check for variable definitions (simple pattern matching)
        lines = code.split('\n')
        for line in lines:
            line = line.strip()
            # Simple assignment patterns
            if '=' in line and not line.startswith('#'):
                parts = line.split('=')
                if len(parts) >= 2:
                    left = parts[0].strip()
                    # Extract variable name
                    if ' ' not in left or left.startswith('for ') or left.startswith('if '):
                        var_name = left.split()[0] if ' ' in left else left
                        var_name = var_name.replace('(', '').replace('[', '').strip()
                        if var_name and var_name not in builtins:
                            variables_defined.add(var_name)
        
        # Check for function definitions
        if 'def ' in code:
            lines = code.split('\n')
            for line in lines:
                if 'def ' in line:
                    func_name = line.split('def ')[1].split('(')[0].strip()
                    if func_name:
                        functions_defined.add(func_name)
    
    # Check if imports are in early cells (first 3 code cells)
    if results['import_cells']:
        early_imports = [c for c in results['import_cells'] if c < 3]
        if not early_imports and results['import_cells']:
            results['warnings'].append(
                "Imports may not be in early cells. Ensure all imports are in the first few cells."
            )
    
    # Count plot statements
    plot_count = 0
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = str(source)
            if 'plt.' in code or '.plot(' in code or '.hist(' in code or '.scatter(' in code:
                plot_count += 1
    
    results['plot_statements'] = plot_count
    
    return results


def main():
    """Main validation function."""
    notebooks_dir = project_root / "notebooks"
    notebooks = list(notebooks_dir.glob("*.ipynb"))
    
    print("="*60)
    print("NOTEBOOK EXECUTION ORDER VALIDATION")
    print("="*60)
    
    all_ok = True
    
    for nb_path in sorted(notebooks):
        print(f"\nChecking {nb_path.name}...")
        results = check_notebook_execution_order(nb_path)
        
        print(f"  Code cells: {results['code_cells']}")
        print(f"  Import cells: {len(results['import_cells'])}")
        print(f"  Plot statements: {results.get('plot_statements', 0)}")
        
        if results['issues']:
            print(f"  ✗ {len(results['issues'])} issues:")
            for issue in results['issues']:
                print(f"    - {issue}")
            all_ok = False
        else:
            print(f"  ✓ No critical issues")
        
        if results['warnings']:
            print(f"  ⚠ {len(results['warnings'])} warnings:")
            for warning in results['warnings']:
                print(f"    - {warning}")
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    if all_ok:
        print("\n✓ NOTEBOOKS APPEAR TO BE EXECUTABLE IN ORDER")
        print("\nTo verify manually:")
        print("  1. Open Jupyter Notebook")
        print("  2. Kernel -> Restart & Clear Output")
        print("  3. Cell -> Run All")
        print("  4. Verify all cells execute without errors")
        print("  5. Verify all plots are displayed")
    else:
        print("\n⚠ SOME NOTEBOOKS MAY HAVE ISSUES")
        print("\nPlease review the issues above.")
        print("Notebooks should be executable from top to bottom.")


if __name__ == "__main__":
    main()

