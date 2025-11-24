"""
Test script for DataLoader class to demonstrate automatic detection
of last valid column and row in DataFrame with trailing empty data.
"""

import pandas as pd
import numpy as np
from processing import DataLoader
import os

def create_test_csv_with_empty_trailing():
    """
    Create a test CSV file with actual data and trailing empty columns/rows.
    """
    # Create sample data: 10 rows x 5 columns of valid numeric data
    data = {
        'Axial_Position': np.arange(0, 10, 1),
        'Caliper_1': np.random.uniform(10, 12, 10),
        'Caliper_2': np.random.uniform(10, 12, 10),
        'Caliper_3': np.random.uniform(10, 12, 10),
        'Caliper_4': np.random.uniform(10, 12, 10),
    }
    
    df = pd.DataFrame(data)
    
    # Add 3 empty columns
    df['Empty_Col_1'] = np.nan
    df['Empty_Col_2'] = np.nan
    df['Empty_Col_3'] = np.nan
    
    # Add 3 empty rows
    empty_rows = pd.DataFrame([[np.nan] * df.shape[1]] * 3, columns=df.columns)
    df = pd.concat([df, empty_rows], ignore_index=True)
    
    # Save to CSV
    test_file = 'test_data_with_empty.csv'
    df.to_csv(test_file, index=False)
    
    return test_file, df

def test_detection():
    """
    Test the automatic detection of last valid column and row.
    """
    print("=" * 70)
    print("Testing DataLoader - Automatic Detection of Valid Data Range")
    print("=" * 70)
    
    # Create test file
    test_file, original_df = create_test_csv_with_empty_trailing()
    
    print("\n1. Original DataFrame shape (with empty trailing data):")
    print(f"   Shape: {original_df.shape} (rows x columns)")
    print(f"   Columns: {list(original_df.columns)}")
    print(f"\n   First 5 rows of original data:")
    print(original_df.head())
    print(f"\n   Last 5 rows of original data (showing empty rows):")
    print(original_df.tail())
    
    # Test DataLoader with automatic detection
    print("\n2. Loading data with DataLoader (automatic detection)...")
    try:
        loader = DataLoader(
            file_path=test_file,
            OD=12.75,
            WT=0.375,
            header_row=0  # First row is header
        )
        
        print(f"\n3. Cleaned DataFrame shape (after automatic detection):")
        print(f"   Shape: {loader._df.shape} (rows x columns)")
        print(f"   Detected last_column index: {loader._last_column}")
        print(f"   Detected last_row index: {loader._last_row}")
        print(f"   Columns: {list(loader._df.columns)}")
        print(f"\n   Cleaned data:")
        print(loader._df)
        
        print("\n4. Validation:")
        expected_rows = 10
        expected_cols = 5
        if loader._df.shape[0] == expected_rows and loader._df.shape[1] == expected_cols:
            print(f"   ✓ SUCCESS: Correctly detected {expected_rows} rows and {expected_cols} columns")
        else:
            print(f"   ✗ FAILED: Expected ({expected_rows}, {expected_cols}), got {loader._df.shape}")
        
        # Check for no NaN values
        if not loader._df.isna().any().any():
            print(f"   ✓ SUCCESS: No missing/NaN values in cleaned data")
        else:
            print(f"   ✗ FAILED: Still contains missing/NaN values")
            
    except Exception as e:
        print(f"\n   ✗ ERROR: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"\n5. Cleanup: Removed test file '{test_file}'")
    
    print("\n" + "=" * 70)

def test_manual_specification():
    """
    Test manual specification of last column and row.
    """
    print("\n" + "=" * 70)
    print("Testing DataLoader - Manual Specification of Data Range")
    print("=" * 70)
    
    # Create test file
    test_file, original_df = create_test_csv_with_empty_trailing()
    
    print("\n1. Testing with manually specified last_column=4 and last_row=9...")
    try:
        loader = DataLoader(
            file_path=test_file,
            OD=12.75,
            WT=0.375,
            header_row=0,
            last_column=4,  # Manually specify
            last_row=9       # Manually specify
        )
        
        print(f"\n2. Result:")
        print(f"   Shape: {loader._df.shape} (rows x columns)")
        print(f"   Specified last_column: 4")
        print(f"   Specified last_row: 9")
        
        if loader._df.shape == (10, 5):
            print(f"   ✓ SUCCESS: Manual specification working correctly")
        else:
            print(f"   ✗ FAILED: Expected (10, 5), got {loader._df.shape}")
            
    except Exception as e:
        print(f"\n   ✗ ERROR: {str(e)}")
    finally:
        # Cleanup
        if os.path.exists(test_file):
            os.remove(test_file)
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_detection()
    test_manual_specification()
    print("\nAll tests completed!")
