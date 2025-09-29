"""
Main Pipeline Script for Rossmann Store Sales Prediction

This script runs the complete data pipeline:
1. Data Exploration
2. Data Cleaning
3. Feature Engineering

Usage:
    python run_pipeline.py --all
    python run_pipeline.py --explore
    python run_pipeline.py --clean
    python run_pipeline.py --engineer
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from data_exploration import DataExplorer
from data_cleaning import DataCleaner
from feature_engineering import FeatureEngineer
from utils import setup_logging, print_section_header


def run_exploration(args):
    """Run data exploration"""
    print_section_header("STEP 1: DATA EXPLORATION")
    
    explorer = DataExplorer(
        train_path=args.train,
        store_path=args.store,
        output_dir=args.figures_dir
    )
    
    explorer.run_full_analysis()


def run_cleaning(args):
    """Run data cleaning"""
    print_section_header("STEP 2: DATA CLEANING")
    
    cleaner = DataCleaner(
        train_path=args.train,
        store_path=args.store,
        test_path=args.test,
        output_dir=args.processed_dir
    )
    
    cleaner.run_cleaning_pipeline()


def run_feature_engineering(args):
    """Run feature engineering"""
    print_section_header("STEP 3: FEATURE ENGINEERING")
    
    engineer = FeatureEngineer(
        train_path=f"{args.processed_dir}/train_cleaned.csv",
        test_path=f"{args.processed_dir}/test_cleaned.csv",
        output_dir=args.processed_dir
    )
    
    engineer.run_feature_engineering()


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Rossmann Store Sales Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run complete pipeline
    python run_pipeline.py --all
    
    # Run individual steps
    python run_pipeline.py --explore
    python run_pipeline.py --clean
    python run_pipeline.py --engineer
    
    # Custom data paths
    python run_pipeline.py --all --train data/raw/train.csv
        """
    )
    
    # Pipeline steps
    parser.add_argument('--all', action='store_true',
                       help='Run complete pipeline (exploration, cleaning, feature engineering)')
    parser.add_argument('--explore', action='store_true',
                       help='Run data exploration only')
    parser.add_argument('--clean', action='store_true',
                       help='Run data cleaning only')
    parser.add_argument('--engineer', action='store_true',
                       help='Run feature engineering only')
    
    # Data paths
    parser.add_argument('--train', type=str, default='data/raw/train.csv',
                       help='Path to train.csv')
    parser.add_argument('--store', type=str, default='data/raw/store.csv',
                       help='Path to store.csv')
    parser.add_argument('--test', type=str, default='data/raw/test.csv',
                       help='Path to test.csv')
    
    # Output directories
    parser.add_argument('--processed-dir', type=str, default='data/processed',
                       help='Directory for processed data')
    parser.add_argument('--figures-dir', type=str, default='reports/figures',
                       help='Directory for figures')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Check if no arguments provided
    if not (args.all or args.explore or args.clean or args.engineer):
        parser.print_help()
        sys.exit(1)
    
    # Run pipeline based on arguments
    print_section_header("ROSSMANN STORE SALES PREDICTION PIPELINE")
    logger.info("Starting pipeline execution...")
    
    try:
        if args.all:
            # Run complete pipeline
            logger.info("Running complete pipeline...")
            run_exploration(args)
            run_cleaning(args)
            run_feature_engineering(args)
            
        else:
            # Run individual steps
            if args.explore:
                run_exploration(args)
            
            if args.clean:
                run_cleaning(args)
            
            if args.engineer:
                run_feature_engineering(args)
        
        print_section_header("PIPELINE EXECUTION COMPLETE")
        logger.info("Pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        raise


if __name__ == '__main__':
    main()
