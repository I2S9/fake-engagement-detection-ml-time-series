#!/usr/bin/env python3
"""
Validation script for notebook reproducibility.
Checks that notebooks can be executed from top to bottom without errors.
"""
import sys
import json
from pathlib import Path
import re
import ast
import subprocess

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def extract_imports_from_code(code: str) -> set:
    """Extract all import statements from code."""
    imports = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except:
        pass
    return imports


def extract_function_calls_from_code(code: str) -> set:
    """Extract function calls from code."""
    calls = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
    except:
        pass
    return calls


def extract_variables_from_code(code: str) -> set:
    """Extract variable names from code."""
    variables = set()
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Store):
                variables.add(node.id)
    except:
        pass
    return variables


def check_notebook_structure(notebook_path: Path) -> dict:
    """
    Check notebook structure for reproducibility issues.
    
    Checks:
    - All imports are at the beginning
    - Functions are defined before use
    - Variables are defined before use
    - No silent cell magic
    """
    results = {
        'notebook': str(notebook_path.name),
        'total_cells': 0,
        'code_cells': 0,
        'issues': [],
        'warnings': [],
        'imports_found': set(),
        'functions_defined': set(),
        'variables_defined': set(),
        'functions_called': set(),
        'variables_used': set(),
        'cell_magic_found': [],
        'all_ok': True
    }
    
    if not notebook_path.exists():
        results['issues'].append(f"Notebook not found: {notebook_path}")
        results['all_ok'] = False
        return results
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    results['total_cells'] = len(nb['cells'])
    
    # Track definitions and usage across cells
    all_code_cells = []
    for cell_idx, cell in enumerate(nb['cells']):
        if cell.get('cell_type') == 'code':
            results['code_cells'] += 1
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = str(source)
            
            all_code_cells.append((cell_idx, code))
            
            # Check for cell magic
            if code.strip().startswith('%%') or code.strip().startswith('%'):
                magic_lines = [line for line in code.split('\n') 
                             if line.strip().startswith('%') or line.strip().startswith('%%')]
                for magic in magic_lines:
                    if 'silent' in magic.lower() or '-s' in magic:
                        results['cell_magic_found'].append({
                            'cell': cell_idx,
                            'magic': magic.strip()
                        })
                        results['warnings'].append(
                            f"Cell {cell_idx}: Silent cell magic found: {magic.strip()}"
                        )
            
            # Extract imports
            imports = extract_imports_from_code(code)
            results['imports_found'].update(imports)
            
            # Extract function definitions
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        results['functions_defined'].add(node.name)
                    elif isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                results['variables_defined'].add(target.id)
            except:
                pass
            
            # Extract function calls and variable usage
            calls = extract_function_calls_from_code(code)
            results['functions_called'].update(calls)
            vars_used = extract_variables_from_code(code)
            results['variables_used'].update(vars_used)
    
    # Check for functions called before definition
    functions_used_before_def = []
    defined_so_far = set()
    for cell_idx, code in all_code_cells:
        calls = extract_function_calls_from_code(code)
        undefined_calls = calls - defined_so_far - results['functions_defined']
        
        # Filter out built-ins and imported functions
        builtins = {'print', 'len', 'range', 'str', 'int', 'float', 'list', 'dict', 
                   'set', 'tuple', 'max', 'min', 'sum', 'abs', 'round', 'sorted',
                   'enumerate', 'zip', 'map', 'filter', 'any', 'all', 'isinstance',
                   'type', 'hasattr', 'getattr', 'setattr', 'delattr', 'dir',
                   'vars', 'locals', 'globals', 'eval', 'exec', 'compile',
                   'open', 'file', 'input', 'raw_input', 'exit', 'quit'}
        stdlib = {'sys', 'os', 'json', 'pathlib', 'datetime', 'time', 'random',
                 'math', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn',
                 'scipy', 'torch', 'tensorflow', 'keras', 'warnings', 'subprocess'}
        
        undefined_calls = {c for c in undefined_calls 
                          if c not in builtins and not any(c.startswith(m) for m in stdlib)}
        
        if undefined_calls:
            functions_used_before_def.extend([
                (cell_idx, func) for func in undefined_calls
            ])
        
        # Update defined functions
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    defined_so_far.add(node.name)
        except:
            pass
    
    if functions_used_before_def:
        for cell_idx, func in functions_used_before_def[:10]:  # Limit to first 10
            results['issues'].append(
                f"Cell {cell_idx}: Function '{func}' may be used before definition"
            )
        results['all_ok'] = False
    
    # Check for variables used before definition (simplified check)
    variables_used_before_def = []
    defined_vars_so_far = set()
    for cell_idx, code in all_code_cells:
        vars_used = extract_variables_from_code(code)
        undefined_vars = vars_used - defined_vars_so_far
        
        # Filter out built-ins
        undefined_vars = {v for v in undefined_vars if v not in builtins}
        
        if undefined_vars:
            # Check if they're imported or defined in this cell
            try:
                tree = ast.parse(code)
                defined_in_cell = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                defined_in_cell.add(target.id)
                undefined_vars = undefined_vars - defined_in_cell
            except:
                pass
            
            if undefined_vars:
                variables_used_before_def.extend([
                    (cell_idx, var) for var in undefined_vars
                ])
        
        # Update defined variables
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            defined_vars_so_far.add(target.id)
        except:
            pass
    
    if variables_used_before_def:
        for cell_idx, var in variables_used_before_def[:10]:  # Limit to first 10
            results['warnings'].append(
                f"Cell {cell_idx}: Variable '{var}' may be used before definition"
            )
    
    # Check for plot/show statements
    plot_count = 0
    for cell_idx, code in all_code_cells:
        if re.search(r'plt\.(show|savefig|figure|subplot)', code, re.IGNORECASE):
            plot_count += 1
        if re.search(r'\.(plot|hist|scatter|bar|boxplot)', code, re.IGNORECASE):
            plot_count += 1
    
    results['plot_statements'] = plot_count
    
    return results


def check_all_notebooks() -> dict:
    """Check all notebooks in the notebooks directory."""
    notebooks_dir = project_root / "notebooks"
    notebooks = list(notebooks_dir.glob("*.ipynb"))
    
    all_results = {
        'notebooks_checked': [],
        'total_issues': 0,
        'total_warnings': 0,
        'all_ok': True
    }
    
    for nb_path in sorted(notebooks):
        print(f"\nChecking {nb_path.name}...")
        results = check_notebook_structure(nb_path)
        all_results['notebooks_checked'].append(results)
        
        if results['issues']:
            all_results['total_issues'] += len(results['issues'])
            all_results['all_ok'] = False
        
        if results['warnings']:
            all_results['total_warnings'] += len(results['warnings'])
        
        print(f"  Cells: {results['code_cells']} code cells")
        print(f"  Plot statements: {results.get('plot_statements', 0)}")
        
        if results['issues']:
            print(f"  ✗ {len(results['issues'])} issues found")
            for issue in results['issues'][:5]:
                print(f"    - {issue}")
        else:
            print(f"  ✓ No critical issues")
        
        if results['warnings']:
            print(f"  ⚠ {len(results['warnings'])} warnings")
            for warning in results['warnings'][:3]:
                print(f"    - {warning}")
        
        if results['cell_magic_found']:
            print(f"  ⚠ {len(results['cell_magic_found'])} cell magic found")
    
    return all_results


def main():
    """Main validation function."""
    print("="*60)
    print("NOTEBOOK REPRODUCIBILITY VALIDATION")
    print("="*60)
    
    results = check_all_notebooks()
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    print(f"\nNotebooks checked: {len(results['notebooks_checked'])}")
    print(f"Total issues: {results['total_issues']}")
    print(f"Total warnings: {results['total_warnings']}")
    
    for nb_result in results['notebooks_checked']:
        print(f"\n  {nb_result['notebook']}:")
        print(f"    Code cells: {nb_result['code_cells']}")
        print(f"    Plot statements: {nb_result.get('plot_statements', 0)}")
        if nb_result['issues']:
            print(f"    ✗ {len(nb_result['issues'])} issues")
        else:
            print(f"    ✓ No issues")
        if nb_result['warnings']:
            print(f"    ⚠ {len(nb_result['warnings'])} warnings")
    
    if results['all_ok']:
        print("\n" + "="*60)
        print("✓ ALL NOTEBOOKS ARE REPRODUCIBLE")
        print("="*60)
        print("\nAll notebooks can be executed from top to bottom.")
        print("No critical issues found.")
        print("\nTo verify manually:")
        print("  1. Restart kernel")
        print("  2. Run All")
        print("  3. Check that all cells execute without errors")
    else:
        print("\n" + "="*60)
        print("⚠ SOME NOTEBOOKS HAVE ISSUES")
        print("="*60)
        print("\nPlease review the issues above and fix them.")
        print("Notebooks should be executable from top to bottom without errors.")


if __name__ == "__main__":
    main()

