#!/usr/bin/env python3
"""
EMBER2024 Ultra-Focused Features Extractor
Extracts ONLY: byteentropy, histogram, label, family, family_confidence, and all entropy values
NO hashes, metadata, or any other fields
"""

import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

def extract_focused_features(input_file, output_file):
    """Extract only specified features from JSONL."""
    print(f"Extracting focused features from {input_file.name}...")
    
    records = []
    total_processed = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                    
                    # Create flattened record with only requested features
                    flattened = {}
                    
                    # Core requested features - NO METADATA/HASHES
                    flattened['label'] = record.get('label')
                    flattened['family'] = record.get('family')
                    flattened['family_confidence'] = record.get('family_confidence')  # This is family_type
                    
                    # Extract histogram (256 features)
                    if 'histogram' in record and isinstance(record['histogram'], list):
                        for i, value in enumerate(record['histogram']):
                            flattened[f'histogram_{i}'] = value
                    else:
                        # Fill with zeros if missing (ensure 256 features)
                        for i in range(256):
                            flattened[f'histogram_{i}'] = 0
                    
                    # Extract byteentropy (256 features)
                    if 'byteentropy' in record and isinstance(record['byteentropy'], list):
                        for i, value in enumerate(record['byteentropy']):
                            flattened[f'byteentropy_{i}'] = value
                    else:
                        # Fill with zeros if missing (ensure 256 features)
                        for i in range(256):
                            flattened[f'byteentropy_{i}'] = 0
                    
                    # Extract entropy values from all subsections
                    # General entropy
                    if 'general' in record and isinstance(record['general'], dict):
                        flattened['general_entropy'] = record['general'].get('entropy')
                    
                    # Strings entropy
                    if 'strings' in record and isinstance(record['strings'], dict):
                        flattened['strings_entropy'] = record['strings'].get('entropy')
                    
                    # Section overlay entropy
                    if 'section' in record and isinstance(record['section'], dict):
                        section_data = record['section']
                        if 'overlay' in section_data and isinstance(section_data['overlay'], dict):
                            flattened['section_overlay_entropy'] = section_data['overlay'].get('entropy')
                    
                    # Check for any other entropy fields in nested structures
                    # This will catch any entropy fields we might have missed
                    def extract_entropy_recursive(obj, prefix=""):
                        """Recursively find all entropy fields"""
                        if isinstance(obj, dict):
                            for key, value in obj.items():
                                if key == 'entropy' and prefix:
                                    field_name = f"{prefix}_entropy"
                                    if field_name not in flattened:  # Don't override already extracted ones
                                        flattened[field_name] = value
                                elif isinstance(value, dict):
                                    new_prefix = f"{prefix}_{key}" if prefix else key
                                    # Skip some known large nested structures
                                    if key not in ['imports', 'string_counts', 'sections']:
                                        extract_entropy_recursive(value, new_prefix)
                    
                    # Extract additional entropy fields
                    extract_entropy_recursive(record)
                    
                    records.append(flattened)
                    total_processed += 1
                    
                    # Process in batches for memory efficiency
                    if len(records) >= 1000:
                        save_batch(records, output_file, total_processed)
                        records = []
                        
                        if total_processed % 10000 == 0:
                            print(f"  Processed {total_processed:,} records...")
                
                except json.JSONDecodeError as e:
                    print(f"  Warning: JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"  Warning: Error processing line {line_num}: {e}")
                    continue
    
    # Handle remaining records
    if records:
        save_batch(records, output_file, total_processed)
    
    print(f"  ✓ Completed: {total_processed:,} records")
    return total_processed

def save_batch(records, output_file, total_processed):
    """Save a batch of records to parquet file."""
    df = pd.DataFrame(records)
    
    if total_processed == len(records):  # First batch
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file, compression='snappy')
    else:  # Append to existing file
        new_table = pa.Table.from_pandas(df)
        existing_table = pq.read_table(output_file)
        combined_table = pa.concat_tables([existing_table, new_table])
        pq.write_table(combined_table, output_file, compression='snappy')

def get_file_category(filename):
    """Determine file category based on filename for proper naming."""
    if 'Win32_train' in filename:
        return 'Win32_train'
    elif 'Win32_test' in filename:
        return 'Win32_test'
    elif 'Win64_train' in filename:
        return 'Win64_train'
    elif 'Win64_test' in filename:
        return 'Win64_test'
    elif 'Dot_Net_train' in filename:
        return 'NET_train'
    elif 'Dot_Net_test' in filename:
        return 'NET_test'
    elif 'APK_train' in filename:
        return 'APK_train'
    elif 'APK_test' in filename:
        return 'APK_test'
    elif 'PDF_train' in filename:
        return 'PDF_train'
    elif 'PDF_test' in filename:
        return 'PDF_test'
    elif 'ELF_train' in filename:
        return 'ELF_train'
    elif 'ELF_test' in filename:
        return 'ELF_test'
    elif 'challenge' in filename:
        return 'challenge'
    else:
        return 'other'

def main():
    input_dir = "a:/Collage/PROJECT/antivirus/ml2/data"
    output_dir = "a:/Collage/PROJECT/antivirus/ml2/ember2024_ultra_focused_features"
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("EMBER2024 Focused Features Extractor")
    print("=" * 60)
    print("Extracting ONLY:")
    print("  • histogram (256 features)")
    print("  • byteentropy (256 features)")
    print("  • label, family, family_confidence")
    print("  • All entropy values from subsections")
    print("  • Skipping all other fields")
    print("=" * 60)
    
    # Get all JSONL files
    input_path = Path(input_dir)
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        print("No JSONL files found in the input directory!")
        return
    
    print(f"Found {len(jsonl_files)} files to process")
    
    # Group files by category for summary
    categories = {}
    for file in jsonl_files:
        category = get_file_category(file.name)
        categories.setdefault(category, []).append(file)
    
    print(f"\nFile distribution:")
    for category, files in sorted(categories.items()):
        print(f"  {category}: {len(files)} files")
    
    # Process each file
    total_records = 0
    successful_conversions = 0
    failed_files = []
    
    for i, input_file in enumerate(sorted(jsonl_files), 1):
        # Create output filename following the same naming schema
        output_file = output_path / f"{input_file.stem}_ultra_focused.parquet"
        
        print(f"\n[{i}/{len(jsonl_files)}] Processing {input_file.name}")
        
        try:
            records = extract_focused_features(input_file, output_file)
            total_records += records
            successful_conversions += 1
            
            # Show file size and create sample
            if records > 0:
                file_size = output_file.stat().st_size / (1024 * 1024)
                print(f"  ✓ Saved: {output_file.name} ({file_size:.1f} MB, {records:,} records)")
                
                # Create sample CSV for the first few files only
                if i <= 3:
                    try:
                        df_sample = pd.read_parquet(output_file).head(2)
                        csv_file = output_path / f"{input_file.stem}_sample.csv"
                        df_sample.to_csv(csv_file, index=False)
                        print(f"  ✓ Sample saved: {csv_file.name}")
                    except Exception as e:
                        print(f"  Warning: Could not create sample CSV: {e}")
            else:
                print(f"  ⚠ Warning: No records extracted from {input_file.name}")
        
        except Exception as e:
            print(f"  ✗ Error processing {input_file.name}: {e}")
            failed_files.append(input_file.name)
            continue
    
    print(f"\n" + "=" * 60)
    print(f"Focused Extraction Complete!")
    print(f"✓ Successfully processed: {successful_conversions}/{len(jsonl_files)} files")
    print(f"✓ Total records extracted: {total_records:,}")
    print(f"✓ Output directory: {output_dir}")
    
    if failed_files:
        print(f"✗ Failed files: {len(failed_files)}")
        for failed in failed_files[:5]:  # Show first 5
            print(f"  - {failed}")
        if len(failed_files) > 5:
            print(f"  ... and {len(failed_files) - 5} more")
    
    # Create comprehensive schema documentation
    schema_info = {
        'extraction_info': {
            'features_extracted': [
                'histogram (256 features: histogram_0 to histogram_255)',
                'byteentropy (256 features: byteentropy_0 to byteentropy_255)',
                'label (target variable)',
                'family (malware family)',
                'family_confidence (family confidence score)',
                'general_entropy (overall file entropy)',
                'strings_entropy (strings section entropy)',
                'section_overlay_entropy (overlay section entropy)',
                'additional entropy fields (dynamically discovered)'
            ],
            'total_core_features': 512,  # 256 histogram + 256 byteentropy
            'target_and_labels': [
                'label', 'family', 'family_confidence'
            ],
            'entropy_fields': [
                'general_entropy', 'strings_entropy', 'section_overlay_entropy',
                'other dynamically discovered entropy fields'
            ],
            'skipped_fields': [
                'md5', 'sha256', 'file_type', 'source_file',
                'imports', 'exports', 'headers', 'sections', 'datadirectories',
                'string_counts', 'behavior', 'file_property', 'packer',
                'exploit', 'group', 'pefilewarnings', 'authenticode details',
                'all other non-entropy fields'
            ]
        },
        'file_summary': {
            'total_files_processed': successful_conversions,
            'total_records': total_records,
            'categories': {k: len(v) for k, v in categories.items()},
            'failed_files': failed_files
        },
        'naming_schema': {
            'input_format': 'YYYY-MM-DD_YYYY-MM-DD_TYPE_SUBSET.jsonl',
            'output_format': 'YYYY-MM-DD_YYYY-MM-DD_TYPE_SUBSET_ultra_focused.parquet',
            'examples': [
                'Input: 2023-09-24_2023-09-30_Win32_train.jsonl',
                'Output: 2023-09-24_2023-09-30_Win32_train_ultra_focused.parquet',
                'Input: 2023-09-24_2023-09-30_challenge_malicious.jsonl',
                'Output: 2023-09-24_2023-09-30_challenge_malicious_ultra_focused.parquet'
            ]
        },
        'dataset_categories': {
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
            'challenge': 'Challenge/malicious samples'
        }
    }
    
    summary_file = output_path / "ultra_focused_extraction_schema.json"
    with open(summary_file, 'w') as f:
        json.dump(schema_info, f, indent=2)
    print(f"✓ Schema documentation saved: {summary_file}")
    
    # Create a detailed column schema file
    if successful_conversions > 0:
        sample_file = list(output_path.glob("*_ultra_focused.parquet"))[0]
        sample_df = pd.read_parquet(sample_file, nrows=1)
        
        # Categorize columns
        column_categories = {
            'identifiers': [],
            'target_labels': [],
            'histogram_features': [],
            'byteentropy_features': [],
            'entropy_features': []
        }
        
        for col in sample_df.columns:
            if col in ['md5', 'sha256', 'file_type', 'source_file']:
                column_categories['identifiers'].append(col)
            elif col in ['label', 'family', 'family_confidence']:
                column_categories['target_labels'].append(col)
            elif col.startswith('histogram_'):
                column_categories['histogram_features'].append(col)
            elif col.startswith('byteentropy_'):
                column_categories['byteentropy_features'].append(col)
            elif 'entropy' in col:
                column_categories['entropy_features'].append(col)
        
        column_schema = {
            'total_columns': len(sample_df.columns),
            'column_categories': column_categories,
            'column_counts': {k: len(v) for k, v in column_categories.items()},
            'data_types': {col: str(dtype) for col, dtype in sample_df.dtypes.items()},
            'all_columns': list(sample_df.columns)
        }
        
        schema_columns_file = output_path / "column_schema.json"
        with open(schema_columns_file, 'w') as f:
            json.dump(column_schema, f, indent=2)
        print(f"✓ Column schema saved: {schema_columns_file}")
        
        # Show summary of what was extracted
        print(f"\n" + "=" * 60)
        print("FEATURE SUMMARY:")
        print(f"  Identifiers: {len(column_categories['identifiers'])} columns")
        print(f"  Target/Labels: {len(column_categories['target_labels'])} columns")
        print(f"  Histogram: {len(column_categories['histogram_features'])} columns")
        print(f"  Byteentropy: {len(column_categories['byteentropy_features'])} columns")
        print(f"  Entropy fields: {len(column_categories['entropy_features'])} columns")
        print(f"  TOTAL: {len(sample_df.columns)} columns")

if __name__ == "__main__":
    main()
