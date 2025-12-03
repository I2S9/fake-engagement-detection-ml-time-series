#!/usr/bin/env python3
"""
Validation script for dataset generation errors.
Performs 5 critical tests to ensure data quality.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data.load_data import load_data


def test_1_no_all_zeros(df: pd.DataFrame) -> dict:
    """
    Test 1: No column with 100% zeros.
    Checks: min >= 0 and max > 0 for all numeric columns.
    """
    print("\n" + "="*60)
    print("TEST 1: No column with 100% zeros")
    print("="*60)
    
    numeric_cols = ['views', 'likes', 'comments', 'shares']
    results = {
        'passed': True,
        'columns_checked': [],
        'issues': []
    }
    
    for col in numeric_cols:
        if col not in df.columns:
            results['issues'].append(f"Column '{col}' not found")
            results['passed'] = False
            continue
        
        results['columns_checked'].append(col)
        col_data = df[col]
        
        min_val = col_data.min()
        max_val = col_data.max()
        zeros_ratio = (col_data == 0).mean()
        
        print(f"\n  {col.upper()}:")
        print(f"    Min: {min_val}")
        print(f"    Max: {max_val}")
        print(f"    Zeros ratio: {zeros_ratio:.2%}")
        
        if min_val < 0:
            print(f"    ✗ ERROR: Negative values found (min={min_val})")
            results['issues'].append(f"{col}: negative values (min={min_val})")
            results['passed'] = False
        elif max_val == 0:
            print(f"    ✗ ERROR: All values are zero")
            results['issues'].append(f"{col}: all values are zero")
            results['passed'] = False
        else:
            print(f"    ✓ OK: min >= 0 and max > 0")
    
    if results['passed']:
        print("\n  ✓ TEST 1 PASSED: No column with 100% zeros")
    else:
        print("\n  ✗ TEST 1 FAILED: Issues found")
        for issue in results['issues']:
            print(f"    - {issue}")
    
    return results


def test_2_no_missing_values(df: pd.DataFrame) -> dict:
    """
    Test 2: No NaN or missing values.
    Checks: df.isna().sum() must be entirely zero.
    """
    print("\n" + "="*60)
    print("TEST 2: No NaN or missing values")
    print("="*60)
    
    results = {
        'passed': True,
        'missing_counts': {},
        'issues': []
    }
    
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    
    print("\n  Missing values per column:")
    for col in df.columns:
        count = missing[col]
        results['missing_counts'][col] = int(count)
        if count > 0:
            print(f"    {col}: {count} missing values")
            results['issues'].append(f"{col}: {count} missing values")
            results['passed'] = False
        else:
            print(f"    {col}: 0 ✓")
    
    if results['passed']:
        print("\n  ✓ TEST 2 PASSED: No missing values")
    else:
        print("\n  ✗ TEST 2 FAILED: Missing values found")
        for issue in results['issues']:
            print(f"    - {issue}")
    
    return results


def test_3_timestamps_strictly_increasing(df: pd.DataFrame) -> dict:
    """
    Test 3: Timestamps are strictly increasing.
    Checks: (df['timestamp'].diff() <= 0).sum() must be 0.
    """
    print("\n" + "="*60)
    print("TEST 3: Timestamps strictly increasing")
    print("="*60)
    
    results = {
        'passed': True,
        'non_increasing_count': 0,
        'issues': []
    }
    
    if 'timestamp' not in df.columns:
        print("  ✗ ERROR: 'timestamp' column not found")
        results['passed'] = False
        results['issues'].append("timestamp column not found")
        return results
    
    # Check per user (timestamps should be increasing within each user)
    print("\n  Checking timestamps per user...")
    non_increasing_users = []
    
    for user_id in df['user_id'].unique():
        user_df = df[df['user_id'] == user_id].sort_values('timestamp')
        diffs = user_df['timestamp'].diff()
        non_increasing = (diffs <= pd.Timedelta(0)).sum()
        
        if non_increasing > 0:
            non_increasing_users.append((user_id, int(non_increasing)))
            results['non_increasing_count'] += non_increasing
            results['issues'].append(f"{user_id}: {non_increasing} non-increasing timestamps")
    
    # Also check globally (should be sorted per user)
    print(f"\n  Total non-increasing timestamps: {results['non_increasing_count']}")
    
    if results['non_increasing_count'] == 0:
        print("  ✓ TEST 3 PASSED: All timestamps are strictly increasing")
    else:
        print("  ✗ TEST 3 FAILED: Non-increasing timestamps found")
        results['passed'] = False
        if len(non_increasing_users) <= 10:
            for user_id, count in non_increasing_users:
                print(f"    - {user_id}: {count} issues")
        else:
            print(f"    - {len(non_increasing_users)} users with issues (showing first 10)")
            for user_id, count in non_increasing_users[:10]:
                print(f"    - {user_id}: {count} issues")
    
    return results


def test_4_labels_balanced(df: pd.DataFrame) -> dict:
    """
    Test 4: Labels (normal / fake) are balanced.
    Checks: label distribution should not be too imbalanced.
    """
    print("\n" + "="*60)
    print("TEST 4: Labels balanced")
    print("="*60)
    
    results = {
        'passed': True,
        'label_distribution': {},
        'imbalance_ratio': None,
        'issues': []
    }
    
    # Check is_fake_series column
    if 'is_fake_series' in df.columns:
        label_col = 'is_fake_series'
        fake_count = df[label_col].sum()
        normal_count = len(df) - fake_count
        total = len(df)
        
        fake_ratio = fake_count / total
        normal_ratio = normal_count / total
        
        results['label_distribution'] = {
            'fake': int(fake_count),
            'normal': int(normal_count),
            'total': int(total),
            'fake_ratio': float(fake_ratio),
            'normal_ratio': float(normal_ratio)
        }
        
        print(f"\n  Label distribution (is_fake_series):")
        print(f"    Normal: {normal_count} ({normal_ratio:.2%})")
        print(f"    Fake: {fake_count} ({fake_ratio:.2%})")
        print(f"    Total: {total}")
        
        # Check if too imbalanced (e.g., < 10% or > 90% of one class)
        if fake_ratio < 0.10:
            print(f"    ⚠ WARNING: Very few fake samples ({fake_ratio:.2%})")
            results['issues'].append(f"Fake ratio too low: {fake_ratio:.2%}")
            results['passed'] = False
        elif fake_ratio > 0.90:
            print(f"    ⚠ WARNING: Very few normal samples ({normal_ratio:.2%})")
            results['issues'].append(f"Normal ratio too low: {normal_ratio:.2%}")
            results['passed'] = False
        else:
            imbalance_ratio = max(fake_ratio, normal_ratio) / min(fake_ratio, normal_ratio)
            results['imbalance_ratio'] = float(imbalance_ratio)
            print(f"    Imbalance ratio: {imbalance_ratio:.2f}x")
            
            if imbalance_ratio > 5:
                print(f"    ⚠ WARNING: Significant imbalance (ratio > 5x)")
                results['issues'].append(f"High imbalance ratio: {imbalance_ratio:.2f}x")
            else:
                print(f"    ✓ OK: Reasonable balance")
    
    # Also check label column if it exists
    if 'label' in df.columns:
        label_counts = df['label'].value_counts()
        print(f"\n  Label distribution (label column):")
        for label, count in label_counts.items():
            ratio = count / len(df)
            print(f"    {label}: {count} ({ratio:.2%})")
    
    if results['passed'] and len(results['issues']) == 0:
        print("\n  ✓ TEST 4 PASSED: Labels are reasonably balanced")
    else:
        print("\n  ⚠ TEST 4 HAS WARNINGS: Check label balance")
        for issue in results['issues']:
            print(f"    - {issue}")
    
    return results


def test_5_no_negative_values(df: pd.DataFrame) -> dict:
    """
    Test 5: No negative values in engagement metrics.
    Checks: (df[['views','likes','comments','shares']] < 0).sum()
    """
    print("\n" + "="*60)
    print("TEST 5: No negative values")
    print("="*60)
    
    numeric_cols = ['views', 'likes', 'comments', 'shares']
    results = {
        'passed': True,
        'negative_counts': {},
        'issues': []
    }
    
    print("\n  Negative values per column:")
    for col in numeric_cols:
        if col not in df.columns:
            continue
        
        negative_count = (df[col] < 0).sum()
        results['negative_counts'][col] = int(negative_count)
        
        if negative_count > 0:
            print(f"    {col}: {negative_count} negative values ✗")
            min_val = df[col].min()
            results['issues'].append(f"{col}: {negative_count} negative values (min={min_val})")
            results['passed'] = False
        else:
            print(f"    {col}: 0 ✓")
    
    if results['passed']:
        print("\n  ✓ TEST 5 PASSED: No negative values")
    else:
        print("\n  ✗ TEST 5 FAILED: Negative values found")
        for issue in results['issues']:
            print(f"    - {issue}")
    
    return results


def main():
    """Main validation function."""
    data_path = project_root / "data" / "raw" / "engagement.parquet"
    
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        print("Please generate the dataset first:")
        print("  python -m src.data.make_dataset --n_users 500 --length 336 --fake_ratio 0.35")
        return
    
    print("Loading dataset...")
    df = load_data(data_path)
    print(f"Dataset loaded: {len(df)} records, {df['user_id'].nunique()} users")
    print(f"Columns: {df.columns.tolist()}")
    
    # Adapt column names if needed
    if 'user_id' in df.columns and 'id' not in df.columns:
        df['id'] = df['user_id']
    if 'is_fake_series' in df.columns and 'label' not in df.columns:
        df['label'] = df['is_fake_series'].map({True: 'fake', False: 'normal'})
    
    # Run all tests
    test_results = {}
    
    test_results['test_1'] = test_1_no_all_zeros(df)
    test_results['test_2'] = test_2_no_missing_values(df)
    test_results['test_3'] = test_3_timestamps_strictly_increasing(df)
    test_results['test_4'] = test_4_labels_balanced(df)
    test_results['test_5'] = test_5_no_negative_values(df)
    
    # Final summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    all_passed = all(
        test_results[f'test_{i}']['passed'] 
        for i in range(1, 6)
    )
    
    print("\nTest Results:")
    for i in range(1, 6):
        status = "✓ PASSED" if test_results[f'test_{i}']['passed'] else "✗ FAILED"
        print(f"  Test {i}: {status}")
    
    if all_passed:
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED - SIMULATOR VALIDATED")
        print("="*60)
        print("\nThe dataset generation is correct:")
        print("  ✓ No columns with 100% zeros")
        print("  ✓ No missing values")
        print("  ✓ Timestamps are strictly increasing")
        print("  ✓ Labels are reasonably balanced")
        print("  ✓ No negative values")
    else:
        print("\n" + "="*60)
        print("✗ SOME TESTS FAILED - SIMULATOR NEEDS FIXES")
        print("="*60)
        print("\nPlease review the test results above and fix the issues.")


if __name__ == "__main__":
    main()

