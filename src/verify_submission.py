import os
import pandas as pd
import numpy as np

def verify_submission(submission_path, test_path):
    print("==================================================")
    print("      STEELGUARD-AI SUBMISSION         ")
    print("==================================================")
    
    # 1. Existence check
    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"[-] ERROR: Submission file does not exist at '{submission_path}'")
        
    df_sub = pd.read_csv(submission_path)
    df_test = pd.read_csv(test_path)
    
    print(f"[+] Loaded submission file with {len(df_sub)} rows.")
    print(f"[+] Loaded test reference file with {len(df_test)} rows.")
    
    errors = 0
    
    # 2. Row count check
    if len(df_sub) != 339:
        print(f"[-] ERROR: Expected exactly 339 rows, but got {len(df_sub)}.")
        errors += 1
    else:
        print("[+] SUCCESS: Row count is exactly 339.")
        
    # 3. Column schema check
    expected_cols = ['CoilID', 'Y']
    if list(df_sub.columns) != expected_cols:
        print(f"[-] ERROR: Columns do not match expected schema {expected_cols}. Got {list(df_sub.columns)}.")
        errors += 1
    else:
        print("[+] SUCCESS: Columns perfectly match expected schema ['CoilID', 'Y'].")
        
    # 4. Null value check
    if df_sub.isnull().any().any():
        null_counts = df_sub.isnull().sum().to_dict()
        print(f"[-] ERROR: Missing/Null values found: {null_counts}")
        errors += 1
    else:
        print("[+] SUCCESS: No missing/null values found.")
        
    # 5. CoilID alignment and sorting check
    if not df_sub['CoilID'].equals(df_test['CoilID']):
        print("[-] ERROR: CoilID values or ordering do not align perfectly with the original test set!")
        # Let's check if the set of CoilIDs is at least correct
        set_sub = set(df_sub['CoilID'])
        set_test = set(df_test['CoilID'])
        if set_sub != set_test:
            print(f"    - CoilIDs set mismatch. Test has {len(set_test)}, Sub has {len(set_sub)}.")
            missing_in_sub = set_test - set_sub
            extra_in_sub = set_sub - set_test
            if missing_in_sub:
                print(f"    - Missing CoilIDs in submission: {list(missing_in_sub)[:5]}")
            if extra_in_sub:
                print(f"    - Unexpected CoilIDs in submission: {list(extra_in_sub)[:5]}")
        else:
            print("    - Note: All CoilIDs are present, but the ORDERING is incorrect!")
        errors += 1
    else:
        print("[+] SUCCESS: CoilID mapping and sorting align perfectly with test reference.")
        
    # 6. Binary schema check
    unique_y = df_sub['Y'].unique()
    invalid_y = [val for val in unique_y if val not in [0, 1]]
    if len(invalid_y) > 0:
        print(f"[-] ERROR: Non-binary predictions found in column 'Y': {invalid_y}")
        errors += 1
    else:
        print("[+] SUCCESS: All prediction values in 'Y' are strictly binary (0 or 1).")
        
    # 7. Label distribution analysis
    defects_count = df_sub['Y'].sum()
    defect_ratio = (defects_count / len(df_sub)) * 100
    print("==================================================")
    print("         SUBMISSION PROFILE & STATISTICS          ")
    print("==================================================")
    print(f"    - Total Samples            : {len(df_sub)}")
    print(f"    - Predicted Normal (0)     : {len(df_sub) - defects_count}")
    print(f"    - Predicted Defective (1)  : {defects_count}")
    print(f"    - Defect Percentage        : {defect_ratio:.2f}%")
    print("==================================================")
    
    if errors == 0:
        print("[SUCCESS] CONGRATULATIONS: Submission file is 100% VALID and ready for hackathon upload!")
        return True
    else:
        print(f"[ERROR] CRITICAL FAILURE: {errors} error(s) must be resolved before submitting!")
        return False

if __name__ == '__main__':
    import sys
    sub_file = os.path.join('submissions', 'expected_submission.csv')
    test_file = os.path.join('data', 'test.csv')
    
    if len(sys.argv) > 1:
        sub_file = sys.argv[1]
    if len(sys.argv) > 2:
        test_file = sys.argv[2]
        
    success = verify_submission(sub_file, test_file)
    if not success:
        sys.exit(1)
