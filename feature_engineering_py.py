"""
Feature Engineering Pipeline for Rossmann Store Sales

This script creates predictive features including:
- Date/time features
- Competition features
- Promotion features
- Lag features
- Rolling window features
- Store-level aggregates
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logging, create_directories, load_dataframe,
    save_dataframe, save_json, print_section_header
)


class FeatureEngineer:
    """Class for feature engineering"""
    
    def __init__(self, train_path, test_path=None, output_dir='data/processed'):
        """
        Initialize FeatureEngineer
        
        Parameters:
        -----------
        train_path : str
            Path to cleaned train.csv
        test_path : str, optional
            Path to cleaned test.csv
        output_dir : str
            Directory for saving feature-engineered data
        """
        self.logger = setup_logging()
        self.output_dir = output_dir
        create_directories([output_dir])
        
        # Load data
        self.logger.info("Loading cleaned datasets...")
        self.train = load_dataframe(train_path, parse_dates=['Date'], logger=self.logger)
        
        if test_path:
            self.test = load_dataframe(test_path, parse_dates=['Date'], logger=self.logger)
        else:
            self.test = None
        
        self.features_created = []
    
    def run_feature_engineering(self):
        """Run complete feature engineering pipeline"""
        self.logger.info("Starting feature engineering pipeline...")
        
        print_section_header("FEATURE ENGINEERING PIPELINE")
        
        # Create features for train
        print_section_header("Engineering Training Features")
        self.train_features = self._engineer_features(self.train, is_train=True)
        
        # Create features for test
        if self.test is not None:
            print_section_header("Engineering Test Features")
            self.test_features = self._engineer_features(self.test, is_train=False)
        
        # Save feature-engineered data
        self.save_features()
        
        self.logger.info("Feature engineering pipeline complete!")
        print_section_header("FEATURE ENGINEERING COMPLETE")
    
    def _engineer_features(self, df, is_train=True):
        """
        Create all features for a dataset
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        is_train : bool
            Whether this is training data
            
        Returns:
        --------
        pd.DataFrame
            Feature-engineered dataframe
        """
        df_features = df.copy()
        
        # Create features
        df_features = self.create_date_features(df_features)
        df_features = self.create_competition_features(df_features)
        df_features = self.create_promotion_features(df_features)
        
        # Only create these for training data
        if is_train:
            df_features = self.create_lag_features(df_features)
            df_features = self.create_rolling_features(df_features)
            df_features = self.create_store_features(df_features)
        
        # Handle missing values from feature creation
        df_features = self._handle_feature_missing_values(df_features)
        
        return df_features
    
    def create_date_features(self, df):
        """
        Create date/time related features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with date features
        """
        print("\n--- Creating Date Features ---")
        
        # Basic date features
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
        df['Quarter'] = df['Date'].dt.quarter
        
        # Day of month features
        df['DayOfMonth'] = df['Date'].dt.day
        df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
        df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
        
        # Week features
        df['IsWeekend'] = (df['DayOfWeek'] >= 6).astype(int)
        
        # Season (meteorological)
        df['Season'] = df['Month'].apply(lambda x:
            1 if x in [12, 1, 2] else
            2 if x in [3, 4, 5] else
            3 if x in [6, 7, 8] else
            4
        )
        
        # Important sales periods
        df['IsChristmasSeason'] = ((df['Month'] == 12) & (df['Day'] >= 15)).astype(int)
        df['IsNewYearWeek'] = ((df['Month'] == 1) & (df['Day'] <= 7)).astype(int)
        df['IsEasterMonth'] = df['Month'].isin([3, 4]).astype(int)
        
        # Payday periods (15th and end of month)
        df['IsPayday'] = ((df['Day'] == 15) | (df['Day'] >= 28)).astype(int)
        
        date_features = ['Year', 'Month', 'Day', 'WeekOfYear', 'Quarter', 'DayOfMonth',
                        'IsMonthStart', 'IsMonthEnd', 'IsWeekend', 'Season',
                        'IsChristmasSeason', 'IsNewYearWeek', 'IsEasterMonth', 'IsPayday']
        
        self.features_created.extend(date_features)
        self.logger.info(f"Created {len(date_features)} date features")
        print(f"✅ Created {len(date_features)} date features")
        
        return df
    
    def create_competition_features(self, df):
        """
        Create competition-related features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with competition features
        """
        print("\n--- Creating Competition Features ---")
        
        # Competition duration in months
        df['CompetitionOpen'] = 12 * (df['Year'] - df['CompetitionOpenSinceYear']) + \
                                (df['Month'] - df['CompetitionOpenSinceMonth'])
        df['CompetitionOpen'] = df['CompetitionOpen'].apply(lambda x: 0 if x < 0 else x)
        
        # Has competition flag
        df['HasCompetition'] = (df['CompetitionDistance'] < 10000).astype(int)
        
        # Log transform of competition distance
        df['CompDistanceLog'] = np.log1p(df['CompetitionDistance'])
        
        comp_features = ['CompetitionOpen', 'HasCompetition', 'CompDistanceLog']
        self.features_created.extend(comp_features)
        self.logger.info(f"Created {len(comp_features)} competition features")
        print(f"✅ Created {len(comp_features)} competition features")
        
        return df
    
    def create_promotion_features(self, df):
        """
        Create promotion-related features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with promotion features
        """
        print("\n--- Creating Promotion Features ---")
        
        # Promo2 duration
        df['Promo2Open'] = 12 * (df['Year'] - df['Promo2SinceYear']) + \
                          (df['WeekOfYear'] - df['Promo2SinceWeek']) / 4
        df['Promo2Open'] = df['Promo2Open'].apply(lambda x: 0 if x < 0 else x)
        
        # Is current month in Promo2 interval?
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                    7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        df['MonthStr'] = df['Month'].map(month_map)
        
        df['IsPromo2Month'] = 0
        for idx in df.index:
            if df.loc[idx, 'PromoInterval'] not in ['None', None]:
                promo_interval = str(df.loc[idx, 'PromoInterval'])
                month_str = df.loc[idx, 'MonthStr']
                if month_str in promo_interval:
                    df.loc[idx, 'IsPromo2Month'] = 1
        
        # Combined promo feature
        df['PromoActive'] = ((df['Promo'] == 1) |
                            ((df['Promo2'] == 1) & (df['IsPromo2Month'] == 1))).astype(int)
        
        # Promo intensity
        df['PromoIntensity'] = df['Promo'] + df['Promo2']
        
        promo_features = ['Promo2Open', 'IsPromo2Month', 'PromoActive', 'PromoIntensity']
        self.features_created.extend(promo_features)
        self.logger.info(f"Created {len(promo_features)} promotion features")
        print(f"✅ Created {len(promo_features)} promotion features")
        
        return df
    
    def create_lag_features(self, df):
        """
        Create lag features (previous sales/customers)
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with lag features
        """
        print("\n--- Creating Lag Features ---")
        
        # Sort by Store and Date
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
        
        # Sales lags
        lag_days = [1, 2, 3, 7, 14, 30]
        for lag in lag_days:
            df[f'Sales_Lag{lag}'] = df.groupby('Store')['Sales'].shift(lag)
        
        # Customer lags
        for lag in [1, 7]:
            df[f'Customers_Lag{lag}'] = df.groupby('Store')['Customers'].shift(lag)
        
        lag_features = [f'Sales_Lag{lag}' for lag in lag_days]
        lag_features.extend(['Customers_Lag1', 'Customers_Lag7'])
        
        self.features_created.extend(lag_features)
        self.logger.info(f"Created {len(lag_features)} lag features")
        print(f"✅ Created {len(lag_features)} lag features")
        
        return df
    
    def create_rolling_features(self, df):
        """
        Create rolling window statistics
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with rolling features
        """
        print("\n--- Creating Rolling Features ---")
        
        # Sort by Store and Date
        df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
        
        # Rolling means
        windows = [7, 14, 30]
        for window in windows:
            # Sales rolling mean
            df[f'Sales_RollingMean{window}'] = df.groupby('Store')['Sales'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
            )
            
            # Sales rolling std
            df[f'Sales_RollingStd{window}'] = df.groupby('Store')['Sales'].transform(
                lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
            )
        
        # Rolling max and min (7-day window)
        df['Sales_RollingMax7'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=1).max()
        )
        df['Sales_RollingMin7'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=7, min_periods=1).min()
        )
        
        rolling_features = []
        for window in windows:
            rolling_features.extend([f'Sales_RollingMean{window}', f'Sales_RollingStd{window}'])
        rolling_features.extend(['Sales_RollingMax7', 'Sales_RollingMin7'])
        
        self.features_created.extend(rolling_features)
        self.logger.info(f"Created {len(rolling_features)} rolling features")
        print(f"✅ Created {len(rolling_features)} rolling features")
        
        return df
    
    def create_store_features(self, df):
        """
        Create store-level aggregate features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with store features
        """
        print("\n--- Creating Store Features ---")
        
        # Store average sales
        store_avg = df.groupby('Store')['Sales'].mean().reset_index()
        store_avg.columns = ['Store', 'Store_AvgSales']
        
        # Store sales std
        store_std = df.groupby('Store')['Sales'].std().reset_index()
        store_std.columns = ['Store', 'Store_SalesStd']
        
        # Store average customers
        store_avg_cust = df.groupby('Store')['Customers'].mean().reset_index()
        store_avg_cust.columns = ['Store', 'Store_AvgCustomers']
        
        # Merge back
        df = df.merge(store_avg, on='Store', how='left')
        df = df.merge(store_std, on='Store', how='left')
        df = df.merge(store_avg_cust, on='Store', how='left')
        
        # Store coefficient of variation
        df['Store_CV'] = df['Store_SalesStd'] / df['Store_AvgSales']
        df['Store_CV'] = df['Store_CV'].replace([np.inf, -np.inf], 0)
        
        # Sales per customer
        df['SalesPerCustomer'] = df['Sales'] / df['Customers']
        df['SalesPerCustomer'] = df['SalesPerCustomer'].replace([np.inf, -np.inf], 0)
        
        store_features = ['Store_AvgSales', 'Store_SalesStd', 'Store_AvgCustomers',
                         'Store_CV', 'SalesPerCustomer']
        
        self.features_created.extend(store_features)
        self.logger.info(f"Created {len(store_features)} store features")
        print(f"✅ Created {len(store_features)} store features")
        
        return df
    
    def _handle_feature_missing_values(self, df):
        """
        Handle missing values created during feature engineering
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
            
        Returns:
        --------
        pd.DataFrame
            Dataframe with missing values filled
        """
        # Fill lag and rolling features with 0 (beginning of time series)
        lag_cols = [col for col in df.columns if 'Lag' in col or 'Rolling' in col]
        if lag_cols:
            df[lag_cols] = df[lag_cols].fillna(0)
            self.logger.info(f"Filled {len(lag_cols)} lag/rolling features with 0")
        
        return df
    
    def save_features(self):
        """Save feature-engineered datasets and metadata"""
        print_section_header("Saving Feature-Engineered Data")
        
        # Save train
        train_path = f"{self.output_dir}/train_features.csv"
        save_dataframe(self.train_features, train_path, self.logger)
        
        # Save test if available
        if self.test is not None:
            test_path = f"{self.output_dir}/test_features.csv"
            save_dataframe(self.test_features, test_path, self.logger)
        
        # Save feature names
        self._save_feature_metadata()
        
        # Save feature summary
        self._save_feature_summary()
    
    def _save_feature_metadata(self):
        """Save feature names and metadata"""
        # Get all feature names (excluding target and identifier columns)
        exclude_cols = ['Date', 'Sales', 'Customers', 'MonthStr']
        feature_names = [col for col in self.train_features.columns 
                        if col not in exclude_cols]
        
        # Save feature names
        feature_path = f"{self.output_dir}/feature_names.json"
        save_json(feature_names, feature_path)
        self.logger.info(f"Saved {len(feature_names)} feature names to {feature_path}")
        
        # Create feature metadata
        feature_metadata = {
            'total_features': len(feature_names),
            'new_features_created': len(self.features_created),
            'feature_categories': {
                'date': len([f for f in self.features_created if any(x in f for x in 
                           ['Year', 'Month', 'Day', 'Week', 'Quarter', 'Season', 'Christmas', 'Easter', 'Payday'])]),
                'competition': len([f for f in self.features_created if 'Comp' in f]),
                'promotion': len([f for f in self.features_created if 'Promo' in f]),
                'lag': len([f for f in self.features_created if 'Lag' in f]),
                'rolling': len([f for f in self.features_created if 'Rolling' in f]),
                'store': len([f for f in self.features_created if 'Store_' in f])
            },
            'feature_names': feature_names
        }
        
        metadata_path = f"{self.output_dir}/feature_metadata.json"
        save_json(feature_metadata, metadata_path)
        self.logger.info(f"Saved feature metadata to {metadata_path}")
    
    def _save_feature_summary(self):
        """Save summary of feature engineering"""
        summary = {
            'total_features': self.train_features.shape[1],
            'new_features_created': len(self.features_created),
            'train_samples': len(self.train_features),
            'test_samples': len(self.test_features) if self.test is not None else 0,
            'date_range_train': f"{self.train_features['Date'].min()} to {self.train_features['Date'].max()}"
        }
        
        print("\n--- Feature Engineering Summary ---")
        for key, value in summary.items():
            print(f"{key}: {value}")
        
        # Save to file
        summary_path = f"{self.output_dir}/feature_summary.txt"
        with open(summary_path, 'w') as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        self.logger.info(f"Saved feature summary to {summary_path}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Feature Engineering Pipeline for Rossmann Store Sales')
    parser.add_argument('--train', type=str, default='data/processed/train_cleaned.csv',
                       help='Path to cleaned train.csv')
    parser.add_argument('--test', type=str, default='data/processed/test_cleaned.csv',
                       help='Path to cleaned test.csv')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory for feature-engineered data')
    
    args = parser.parse_args()
    
    # Run feature engineering
    engineer = FeatureEngineer(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output
    )
    
    engineer.run_feature_engineering()


if __name__ == '__main__':
    main()
