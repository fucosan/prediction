#!/usr/bin/env python3
"""
Command line interface for the data mapper module
"""

import argparse
import pandas as pd
import os
import sys
from datetime import datetime

from src.data_mapper import (
    create_mapper, update_mapper, load_mapper, 
    filter_by_mapper, validate_data_against_mapper,
    backup_mapper, restore_mapper, get_mapper_stats
)
from src.data_mapper.utils import list_backups

from config.config import DATA_MAPPER_PATH

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Data Mapper Management Tool')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a new mapper from data')
    create_parser.add_argument('input_file', help='Input data file (CSV or Excel)')
    create_parser.add_argument('--output', help='Output mapper path', default=DATA_MAPPER_PATH)
    create_parser.add_argument('--overwrite', action='store_true', help='Overwrite existing mapper')
    create_parser.add_argument('--add-cols', nargs='+', help='Additional columns to include')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update an existing mapper')
    update_parser.add_argument('input_file', help='Input data file with new data')
    update_parser.add_argument('--mapper', help='Mapper file to update', default=DATA_MAPPER_PATH)
    update_parser.add_argument('--mode', choices=['merge', 'replace', 'add_only', 'deactivate_missing'],
                            default='merge', help='Update mode')
    update_parser.add_argument('--no-backup', action='store_true', help='Skip creating backup')
    
    # Filter command
    filter_parser = subparsers.add_parser('filter', help='Filter data using mapper')
    filter_parser.add_argument('input_file', help='Input data file to filter')
    filter_parser.add_argument('--output', help='Output file for filtered data')
    filter_parser.add_argument('--mapper', help='Mapper file to use', default=DATA_MAPPER_PATH)
    filter_parser.add_argument('--include-inactive', action='store_true', help='Include inactive entries')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate data against mapper')
    validate_parser.add_argument('input_file', help='Input data file to validate')
    validate_parser.add_argument('--mapper', help='Mapper file to use', default=DATA_MAPPER_PATH)
    validate_parser.add_argument('--output', help='Output file for validation report')
    validate_parser.add_argument('--include-inactive', action='store_true', help='Include inactive entries')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Get mapper statistics')
    stats_parser.add_argument('--mapper', help='Mapper file', default=DATA_MAPPER_PATH)
    
    # Backup command
    backup_parser = subparsers.add_parser('backup', help='Backup mapper')
    backup_parser.add_argument('--mapper', help='Mapper file to backup', default=DATA_MAPPER_PATH)
    
    # List backups command
    list_backups_parser = subparsers.add_parser('list-backups', help='List available backups')
    
    # Restore command
    restore_parser = subparsers.add_parser('restore', help='Restore mapper from backup')
    restore_parser.add_argument('backup_file', help='Backup file to restore from')
    restore_parser.add_argument('--no-backup', action='store_true', help='Skip backing up current mapper')
    
    return parser.parse_args()

def load_data(file_path):
    """Load data from file"""
    print(f"Loading data from {file_path}")
    
    if file_path.endswith('.csv'):
        return pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def main():
    """Main function"""
    args = parse_arguments()
    
    if not args.command:
        print("No command specified. Use --help for usage information.")
        sys.exit(1)
    
    start_time = datetime.now()
    print(f"=== Starting Data Mapper {args.command} at {start_time} ===")
    
    try:
        if args.command == 'create':
            # Create a new mapper
            data = load_data(args.input_file)
            mapper = create_mapper(
                data,
                output_path=args.output,
                additional_cols=args.add_cols,
                overwrite=args.overwrite
            )
            print(f"Created mapper with {len(mapper)} entries at {args.output}")
        
        elif args.command == 'update':
            # Update existing mapper
            data = load_data(args.input_file)
            mapper, stats = update_mapper(
                data,
                mapper_path=args.mapper,
                mode=args.mode,
                backup=not args.no_backup
            )
            print(f"Updated mapper at {args.mapper}")
            print(f"Update stats: {stats}")
        
        elif args.command == 'filter':
            # Filter data using mapper
            data = load_data(args.input_file)
            filtered_data, stats = filter_by_mapper(
                data,
                mapper_path=args.mapper,
                active_only=not args.include_inactive,
                return_stats=True
            )
            
            if args.output:
                if args.output.endswith('.csv'):
                    filtered_data.to_csv(args.output, index=False)
                elif args.output.endswith(('.xls', '.xlsx')):
                    filtered_data.to_excel(args.output, index=False)
                else:
                    # Default to CSV
                    filtered_data.to_csv(args.output, index=False)
                print(f"Filtered data saved to {args.output}")
            else:
                # Print summary if no output file specified
                print("\nFiltering Summary:")
                print(f"Original records: {stats['before']}")
                print(f"Filtered records: {stats['after']}")
                print(f"Removed records: {stats['removed']} ({stats['removed_percent']}%)")
        
        elif args.command == 'validate':
            # Validate data against mapper
            data = load_data(args.input_file)
            result = validate_data_against_mapper(
                data,
                mapper_path=args.mapper,
                active_only=not args.include_inactive
            )
            
            # Print summary
            print("\nValidation Summary:")
            print(f"Total rows: {result['total_rows']}")
            print(f"Unique combinations: {result['unique_combinations']}")
            print(f"Valid combinations: {result['valid_combinations']} ({result['valid_percent']}%)")
            print(f"Invalid combinations: {result['invalid_combinations']}")
            
            if result['invalid_combinations'] > 0:
                print("\nTop 10 invalid entries:")
                sorted_invalid = sorted(
                    result['invalid_entries'].items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:10]
                
                for entry, count in sorted_invalid:
                    print(f"  {entry}: {count} occurrences")
            
            # Save report if requested
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"\nDetailed validation report saved to {args.output}")
        
        elif args.command == 'stats':
            # Get mapper statistics
            stats = get_mapper_stats(args.mapper)
            
            if not stats['exists']:
                print(f"Mapper not found: {args.mapper}")
                sys.exit(1)
            
            print("\nMapper Statistics:")
            print(f"File path: {args.mapper}")
            print(f"File size: {stats['file_size_bytes']} bytes")
            print(f"File created: {stats['file_created']}")
            print(f"File modified: {stats['file_modified']}")
            print(f"Total entries: {stats['total_entries']}")
            print(f"Unique sites: {stats['unique_sites']}")
            print(f"Unique items: {stats['unique_items']}")
            
            if 'active_entries' in stats:
                print(f"Active entries: {stats['active_entries']} ({stats['active_percent']}%)")
                print(f"Inactive entries: {stats['inactive_entries']}")
            
            if 'latest_update' in stats:
                print(f"Latest update: {stats['latest_update']}")
                print(f"Earliest update: {stats['earliest_update']}")
        
        elif args.command == 'backup':
            # Backup mapper
            backup_path = backup_mapper(args.mapper)
            print(f"Mapper backed up to {backup_path}")
        
        elif args.command == 'list-backups':
            # List available backups
            backups = list_backups()
            
            if not backups:
                print("No backups found.")
                sys.exit(0)
            
            print(f"\nFound {len(backups)} backups:")
            for i, backup in enumerate(backups, 1):
                print(f"{i}. {backup['filename']}")
                print(f"   Created: {backup['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Size: {backup['size_bytes']} bytes")
                print(f"   Entries: {backup['row_count']}")
                print(f"   Path: {backup['path']}")
                print()
        
        elif args.command == 'restore':
            # Restore mapper from backup
            restored_path = restore_mapper(
                args.backup_file,
                make_backup=not args.no_backup
            )
            print(f"Mapper restored to {restored_path}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    end_time = datetime.now()
    print(f"=== Operation completed in {end_time - start_time} ===")

if __name__ == "__main__":
    main()