"""
Exploratory Data Analysis for Rossmann Store Sales

This script performs comprehensive EDA including:
- Data overview and statistics
- Sales analysis
- Store performance analysis
- Temporal patterns
- Promotion effectiveness
- Competition impact
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logging, create_directories, load_dataframe, 
    save_plot, print_section_header, calculate_missing_stats
)


class DataExplorer:
    """Class for exploratory data analysis"""
    
    def __init__(self, train_path, store_path, output_dir='reports/figures'):
        """
        Initialize DataExplorer
        
        Parameters:
        -----------
        train_path : str
            Path to train.csv
        store_path : str
            Path to store.csv
        output_dir : str
            Directory for saving plots
        """
        self.logger = setup_logging()
        self.output_dir = output_dir
        create_directories([output_dir])
        
        # Load data
        self.logger.info("Loading datasets...")
        self.train = load_dataframe(train_path, parse_dates=['Date'], logger=self.logger)
        self.store = load_dataframe(store_path, logger=self.logger)
        
        # Merge train with store info
        self.df = self.train.merge(self.store, on='Store', how='left')
        self.logger.info(f"Merged dataset shape: {self.df.shape}")
        
        # Filter to open stores only for analysis
        self.df_open = self.df[self.df['Open'] == 1].copy()
        self.logger.info(f"Open stores dataset shape: {self.df_open.shape}")
    
    def run_full_analysis(self):
        """Run complete exploratory data analysis"""
        self.logger.info("Starting full exploratory data analysis...")
        
        print_section_header("ROSSMANN STORE SALES - EXPLORATORY DATA ANALYSIS")
        
        # Run all analysis modules
        self.data_overview()
        self.sales_analysis()
        self.store_analysis()
        self.temporal_analysis()
        self.promotion_analysis()
        self.competition_analysis()
        self.correlation_analysis()
        
        self.logger.info("Exploratory data analysis complete!")
        print_section_header("ANALYSIS COMPLETE")
    
    def data_overview(self):
        """Generate data overview and summary statistics"""
        print_section_header("Data Overview")
        
        # Dataset info
        print(f"Dataset Shape: {self.df.shape}")
        print(f"Date Range: {self.df['Date'].min()} to {self.df['Date'].max()}")
        print(f"Number of Stores: {self.df['Store'].nunique()}")
        print(f"Total Records: {len(self.df):,}")
        print(f"Open Store Records: {len(self.df_open):,}")
        
        # Missing values
        print("\nMissing Values:")
        missing = calculate_missing_stats(self.df)
        if len(missing) > 0:
            print(missing)
        else:
            print("No missing values found!")
        
        # Summary statistics
        print("\nSales Statistics (Open Stores):")
        print(self.df_open['Sales'].describe())
        
        print("\nCustomers Statistics (Open Stores):")
        print(self.df_open['Customers'].describe())
    
    def sales_analysis(self):
        """Analyze sales distributions and patterns"""
        print_section_header("Sales Analysis")
        
        df = self.df_open
        
        # Calculate statistics
        mean_sales = df['Sales'].mean()
        median_sales = df['Sales'].median()
        
        print(f"Mean Sales: €{mean_sales:.2f}")
        print(f"Median Sales: €{median_sales:.2f}")
        print(f"Total Sales: €{df['Sales'].sum():,.0f}")
        
        # Sales distribution plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Sales distribution
        axes[0].hist(df['Sales'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(mean_sales, color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: €{mean_sales:.0f}')
        axes[0].axvline(median_sales, color='green', linestyle='--', linewidth=2,
                       label=f'Median: €{median_sales:.0f}')
        axes[0].set_xlabel('Sales (€)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Distribution of Daily Sales', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Customers distribution
        axes[1].hist(df['Customers'], bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Number of Customers', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Distribution of Daily Customers', fontsize=14, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        save_plot(fig, 'sales_distribution.png', self.output_dir)
        self.logger.info("Saved sales distribution plot")
        
        # Sales per customer
        df['SalesPerCustomer'] = df['Sales'] / df['Customers']
        avg_transaction = df['SalesPerCustomer'].mean()
        
        print(f"\nAverage Transaction Value: €{avg_transaction:.2f}")
    
    def store_analysis(self):
        """Analyze store performance and characteristics"""
        print_section_header("Store Analysis")
        
        df = self.df_open
        
        # Store type analysis
        print("Sales by Store Type:")
        store_type_stats = df.groupby('StoreType')['Sales'].agg(['mean', 'median', 'count'])
        print(store_type_stats)
        
        # Assortment analysis
        print("\nSales by Assortment:")
        assortment_stats = df.groupby('Assortment')['Sales'].agg(['mean', 'median', 'count'])
        print(assortment_stats)
        
        # Top and bottom stores
        store_performance = df.groupby('Store')['Sales'].mean().sort_values(ascending=False)
        print(f"\nTop 5 Performing Stores:")
        print(store_performance.head())
        print(f"\nBottom 5 Performing Stores:")
        print(store_performance.tail())
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Store type
        store_type_sales = df.groupby('StoreType')['Sales'].mean()
        axes[0].bar(store_type_sales.index, store_type_sales.values, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Store Type', fontsize=12)
        axes[0].set_ylabel('Average Sales (€)', fontsize=12)
        axes[0].set_title('Average Sales by Store Type', fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Assortment
        assortment_sales = df.groupby('Assortment')['Sales'].mean()
        axes[1].bar(assortment_sales.index, assortment_sales.values, color='coral', alpha=0.7)
        axes[1].set_xlabel('Assortment Type', fontsize=12)
        axes[1].set_ylabel('Average Sales (€)', fontsize=12)
        axes[1].set_title('Average Sales by Assortment', fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_plot(fig, 'store_analysis.png', self.output_dir)
        self.logger.info("Saved store analysis plot")
    
    def temporal_analysis(self):
        """Analyze temporal patterns in sales"""
        print_section_header("Temporal Analysis")
        
        df = self.df_open.copy()
        
        # Day of week analysis
        dow_mapping = {1: 'Mon', 2: 'Tue', 3: 'Wed', 4: 'Thu', 5: 'Fri', 6: 'Sat', 7: 'Sun'}
        df['DayName'] = df['DayOfWeek'].map(dow_mapping)
        
        print("Average Sales by Day of Week:")
        dow_sales = df.groupby('DayName')['Sales'].mean().reindex(
            ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        )
        print(dow_sales)
        
        # Monthly analysis
        df['Month'] = df['Date'].dt.month
        print("\nAverage Sales by Month:")
        monthly_sales = df.groupby('Month')['Sales'].mean()
        print(monthly_sales)
        
        # Visualization
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Day of week
        dow_sales.plot(kind='bar', ax=axes[0], color='teal', alpha=0.7)
        axes[0].set_xlabel('Day of Week', fontsize=12)
        axes[0].set_ylabel('Average Sales (€)', fontsize=12)
        axes[0].set_title('Average Sales by Day of Week', fontsize=14, fontweight='bold')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Monthly
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[1].bar(range(1, 13), monthly_sales.values, color='indianred', alpha=0.7)
        axes[1].set_xlabel('Month', fontsize=12)
        axes[1].set_ylabel('Average Sales (€)', fontsize=12)
        axes[1].set_title('Average Sales by Month', fontsize=14, fontweight='bold')
        axes[1].set_xticks(range(1, 13))
        axes[1].set_xticklabels(month_names)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_plot(fig, 'temporal_patterns.png', self.output_dir)
        self.logger.info("Saved temporal patterns plot")
        
        # Time series plot
        fig, ax = plt.subplots(figsize=(15, 6))
        daily_sales = df.groupby('Date')['Sales'].mean()
        ax.plot(daily_sales.index, daily_sales.values, color='navy', alpha=0.6, linewidth=1)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Average Sales (€)', fontsize=12)
        ax.set_title('Average Daily Sales Over Time', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        save_plot(fig, 'sales_timeseries.png', self.output_dir)
        self.logger.info("Saved time series plot")
    
    def promotion_analysis(self):
        """Analyze promotion effectiveness"""
        print_section_header("Promotion Analysis")
        
        df = self.df_open
        
        # Promotion statistics
        promo_stats = df.groupby('Promo')['Sales'].agg(['mean', 'median', 'count'])
        print("Sales Statistics by Promotion:")
        print(promo_stats)
        
        # Calculate lift
        sales_no_promo = df[df['Promo'] == 0]['Sales'].mean()
        sales_with_promo = df[df['Promo'] == 1]['Sales'].mean()
        promo_lift = ((sales_with_promo - sales_no_promo) / sales_no_promo) * 100
        
        print(f"\nPromotion Lift: {promo_lift:.2f}%")
        print(f"Sales without promo: €{sales_no_promo:.2f}")
        print(f"Sales with promo: €{sales_with_promo:.2f}")
        
        # Visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sales comparison
        promo_sales = df.groupby('Promo')['Sales'].mean()
        axes[0].bar(['No Promo', 'Promo'], promo_sales.values, 
                   color=['lightcoral', 'lightgreen'], alpha=0.7)
        axes[0].set_ylabel('Average Sales (€)', fontsize=12)
        axes[0].set_title('Average Sales: With vs Without Promotion', 
                         fontsize=14, fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        
        # Customer comparison
        promo_customers = df.groupby('Promo')['Customers'].mean()
        axes[1].bar(['No Promo', 'Promo'], promo_customers.values,
                   color=['lightblue', 'orange'], alpha=0.7)
        axes[1].set_ylabel('Average Customers', fontsize=12)
        axes[1].set_title('Average Customers: With vs Without Promotion',
                         fontsize=14, fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        save_plot(fig, 'promotion_analysis.png', self.output_dir)
        self.logger.info("Saved promotion analysis plot")
    
    def competition_analysis(self):
        """Analyze competition impact"""
        print_section_header("Competition Analysis")
        
        df = self.df_open[self.df_open['CompetitionDistance'].notna()].copy()
        
        # Create distance bins
        df['CompDistanceBin'] = pd.cut(
            df['CompetitionDistance'],
            bins=[0, 500, 1000, 2000, 5000, 50000],
            labels=['<500m', '500-1km', '1-2km', '2-5km', '>5km']
        )
        
        print("Sales by Competition Distance:")
        comp_sales = df.groupby('CompDistanceBin')['Sales'].mean()
        print(comp_sales)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        comp_sales.plot(kind='bar', ax=ax, color='darkorange', alpha=0.7)
        ax.set_xlabel('Competition Distance', fontsize=12)
        ax.set_ylabel('Average Sales (€)', fontsize=12)
        ax.set_title('Impact of Competition Distance on Sales', 
                    fontsize=14, fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        save_plot(fig, 'competition_analysis.png', self.output_dir)
        self.logger.info("Saved competition analysis plot")
    
    def correlation_analysis(self):
        """Analyze correlations between features"""
        print_section_header("Correlation Analysis")
        
        df = self.df_open
        
        # Select numerical columns
        corr_cols = ['Sales', 'Customers', 'Promo', 'CompetitionDistance', 'DayOfWeek']
        corr_matrix = df[corr_cols].corr()
        
        print("Correlation Matrix:")
        print(corr_matrix.round(3))
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_plot(fig, 'correlation_matrix.png', self.output_dir)
        self.logger.info("Saved correlation matrix plot")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Exploratory Data Analysis for Rossmann Store Sales')
    parser.add_argument('--train', type=str, default='data/raw/train.csv',
                       help='Path to train.csv')
    parser.add_argument('--store', type=str, default='data/raw/store.csv',
                       help='Path to store.csv')
    parser.add_argument('--output', type=str, default='reports/figures',
                       help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Run analysis
    explorer = DataExplorer(
        train_path=args.train,
        store_path=args.store,
        output_dir=args.output
    )
    
    explorer.run_full_analysis()


if __name__ == '__main__':
    main()
