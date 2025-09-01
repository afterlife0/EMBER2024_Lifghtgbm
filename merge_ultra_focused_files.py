#!/usr/bin/env python3
"""
EMBER2024 Ultra-Focused Files Merger
Merges all ultra-focused parquet files by category (train/test) and file type
"""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import json
from collections import defaultdict
import time

def get_file_categories(input_dir):
    """Categorize files by type and subset"""
    categories = defaultdict(list)
    
    for file in Path(input_dir).glob("*_ultra_focused.parquet"):
        filename = file.stem.replace("_ultra_focused", "")
        
        # Extract the file type and subset from filename
        # Format: YYYY-MM-DD_YYYY-MM-DD_TYPE_SUBSET
        parts = filename.split("_")
        if len(parts) >= 4:
            # Get the last two parts (TYPE_SUBSET)
            file_type = parts[-2]
            subset = parts[-1]
            
            # Special handling for challenge files
            if "challenge" in filename:
                file_type = "challenge"
                subset = "malicious"
            
            category_key = f"{file_type}_{subset}"
            categories[category_key].append(file)
    
    return categories

def merge_category_files(files, output_file, category_name):
    """Merge multiple parquet files into one"""
    print(f"\n[{category_name}] Merging {len(files)} files...")
    
    dfs = []
    total_records = 0
    
    for i, file in enumerate(files, 1):
        print(f"  Reading file {i}/{len(files)}: {file.name}")
        try:
            df = pd.read_parquet(file)
            dfs.append(df)
            total_records += len(df)
            print(f"    ✓ {len(df):,} records")
        except Exception as e:
            print(f"    ✗ Error reading {file.name}: {e}")
            continue
    
    if not dfs:
        print(f"  ✗ No valid files found for {category_name}")
        return 0
    
    print(f"  Concatenating {len(dfs)} dataframes...")
    merged_df = pd.concat(dfs, ignore_index=True)
    
    print(f"  Saving merged file: {output_file.name}")
    merged_df.to_parquet(output_file, compression='snappy', index=False)
    
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {total_records:,} records, {file_size_mb:.1f} MB")
    
    return total_records

def main():
    print("EMBER2024 Ultra-Focused Files Merger")
    print("=" * 60)
    
    input_dir = "a:/Collage/PROJECT/antivirus/ml2/ember2024_ultra_focused_features"
    output_dir = Path("a:/Collage/PROJECT/antivirus/ml2/ember2024_merged_categories")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Get file categories
    categories = get_file_categories(input_dir)
    
    print(f"Found {len(categories)} categories:")
    for category, files in categories.items():
        print(f"  {category}: {len(files)} files")
    print()
    
    # Merge each category
    merged_stats = {}
    total_files_processed = 0
    total_records = 0
    
    start_time = time.time()
    
    for category, files in sorted(categories.items()):
        output_file = output_dir / f"ember2024_{category}_merged.parquet"
        
        records_count = merge_category_files(files, output_file, category)
        
        merged_stats[category] = {
            'files_merged': len(files),
            'total_records': records_count,
            'output_file': output_file.name,
            'file_size_mb': round(output_file.stat().st_size / (1024 * 1024), 1) if output_file.exists() else 0
        }
        
        total_files_processed += len(files)
        total_records += records_count
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    print("\n" + "=" * 60)
    print("Merging Complete!")
    print("=" * 60)
    
    print(f"✓ Categories processed: {len(categories)}")
    print(f"✓ Total files merged: {total_files_processed}")
    print(f"✓ Total records: {total_records:,}")
    print(f"✓ Processing time: {processing_time:.1f} seconds")
    print(f"✓ Output directory: {output_dir}")
    
    # Show merged file details
    print("\nMerged Files Summary:")
    print("-" * 60)
    for category, stats in sorted(merged_stats.items()):
        print(f"{category:25s} | {stats['files_merged']:3d} files → {stats['total_records']:8,} records | {stats['file_size_mb']:6.1f} MB")
    
    # Create summary JSON
    summary = {
        'merger_info': {
            'total_categories': len(categories),
            'total_files_processed': total_files_processed,
            'total_records': total_records,
            'processing_time_seconds': round(processing_time, 1),
            'output_directory': str(output_dir)
        },
        'merged_categories': merged_stats,
        'category_descriptions': {
            'Win32_train': 'Windows 32-bit training data',
            'Win32_test': 'Windows 32-bit test data',
            'Win64_train': 'Windows 64-bit training data', 
            'Win64_test': 'Windows 64-bit test data',
            'NET_train': '.NET training data',
            'NET_test': '.NET test data',
            'APK_train': 'Android APK training data',
            'APK_test': 'Android APK test data',
            'PDF_train': 'PDF training data',
            'PDF_test': 'PDF test data',
            'ELF_train': 'Linux ELF training data',
            'ELF_test': 'Linux ELF test data',
            'challenge_malicious': 'Challenge/malicious samples'
        }
    }
    
    summary_file = output_dir / "merger_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved: {summary_file}")
    
    # Quick verification
    print("\nQuick Verification:")
    print("-" * 60)
    sample_file = None
    for category, stats in merged_stats.items():
        if stats['total_records'] > 0:
            sample_file = output_dir / stats['output_file']
            break
    
    if sample_file and sample_file.exists():
        df_sample = pd.read_parquet(sample_file)
        df_sample = df_sample.head(1)  # Get just first row for verification
        print(f"Sample file: {sample_file.name}")
        print(f"Columns: {len(df_sample.columns)}")
        print(f"Sample columns: {list(df_sample.columns[:10])}...")  # Show first 10 columns
        print("✓ Merged files ready for machine learning!")
    
    print(f"\n🎉 All files successfully merged by category!")

if __name__ == "__main__":
    main()
