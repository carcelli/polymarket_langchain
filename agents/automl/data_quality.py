"""
Data Quality Validation and Cleaning for Polymarket ML

Ensures data quality, handles missing values, and validates
ML-ready datasets for reliable model training.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from scipy import stats

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """
    Comprehensive data quality validation for ML datasets.

    Handles:
    - Missing value detection and imputation
    - Outlier detection and treatment
    - Feature distribution analysis
    - Data integrity validation
    - Statistical quality checks
    """

    def __init__(self):
        self.imputers = {}
        self.scalers = {}
        self.encoders = {}

    def validate_dataset_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate the overall structure and quality of the dataset.

        Args:
            df: Dataset to validate

        Returns:
            Validation report
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_data': {},
            'data_types': {},
            'duplicate_rows': df.duplicated().sum(),
            'issues': []
        }

        # Check for required columns
        required_columns = [
            'market_id', 'question', 'category', 'yes_price', 'volume',
            'resolved', 'actual_outcome'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            report['issues'].append(f"Missing required columns: {missing_columns}")

        # Analyze missing data
        for col in df.columns:
            missing_pct = df[col].isnull().sum() / len(df) * 100
            report['missing_data'][col] = missing_pct

            if missing_pct > 50:
                report['issues'].append(f"High missing data in {col}: {missing_pct:.1f}%")
            elif missing_pct > 20:
                report['issues'].append(f"Moderate missing data in {col}: {missing_pct:.1f}%")

        # Data type analysis
        for col in df.columns:
            report['data_types'][col] = str(df[col].dtype)

        # Check for data integrity issues
        if 'yes_price' in df.columns:
            invalid_prices = ((df['yes_price'] < 0) | (df['yes_price'] > 1)).sum()
            if invalid_prices > 0:
                report['issues'].append(f"Invalid price data: {invalid_prices} rows")

        if 'volume' in df.columns:
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                report['issues'].append(f"Negative volume: {negative_volume} rows")

        # Target variable validation
        if 'will_resolve_yes' in df.columns:
            target_dist = df['will_resolve_yes'].value_counts()
            report['target_distribution'] = target_dist.to_dict()

            if len(target_dist) < 2:
                report['issues'].append("Target variable has only one class")

        return report

    def detect_outliers(self, df: pd.DataFrame, columns: List[str] = None,
                       method: str = 'iqr') -> Dict[str, Any]:
        """
        Detect outliers in numerical columns.

        Args:
            df: Dataset to analyze
            columns: Columns to check (default: numerical columns)
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')

        Returns:
            Outlier detection report
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_report = {}

        for col in columns:
            if col not in df.columns:
                continue

            values = df[col].dropna()

            if len(values) < 10:
                continue

            if method == 'iqr':
                # IQR method
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = ((values < lower_bound) | (values > upper_bound)).sum()
                outlier_pct = outliers / len(values) * 100

            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs(stats.zscore(values))
                outliers = (z_scores > 3).sum()
                outlier_pct = outliers / len(values) * 100

            outlier_report[col] = {
                'outlier_count': outliers,
                'outlier_percentage': outlier_pct,
                'method': method
            }

        return outlier_report

    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """
        Handle missing values in the dataset.

        Args:
            df: Dataset with missing values
            strategy: Imputation strategy ('auto', 'mean', 'median', 'most_frequent')

        Returns:
            Dataset with imputed values
        """
        df_clean = df.copy()

        # Handle different column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        # Numeric imputation
        if strategy == 'auto':
            # Use median for skewed data, mean for normal
            for col in numeric_cols:
                if df[col].isnull().sum() > 0:
                    skewness = df[col].skew()
                    if abs(skewness) > 1:
                        impute_strategy = 'median'
                    else:
                        impute_strategy = 'mean'
                else:
                    continue

                imputer = SimpleImputer(strategy=impute_strategy)
                df_clean[col] = imputer.fit_transform(df[[col]]).ravel()
                self.imputers[col] = imputer

        # Categorical imputation
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                imputer = SimpleImputer(strategy='most_frequent')
                df_clean[col] = imputer.fit_transform(df[[col]]).ravel()
                self.imputers[col] = imputer

        logger.info(f"Imputed missing values in {len(self.imputers)} columns")
        return df_clean

    def remove_outliers(self, df: pd.DataFrame, columns: List[str] = None,
                       method: str = 'iqr', threshold: float = 0.05) -> pd.DataFrame:
        """
        Remove outlier rows from the dataset.

        Args:
            df: Dataset to clean
            columns: Columns to check for outliers
            method: Outlier detection method
            threshold: Maximum allowed outlier percentage per column

        Returns:
            Dataset with outliers removed
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        outlier_report = self.detect_outliers(df, columns, method)

        # Find columns with too many outliers
        problematic_cols = [
            col for col, stats in outlier_report.items()
            if stats['outlier_percentage'] > threshold * 100
        ]

        if not problematic_cols:
            logger.info("No significant outliers detected")
            return df

        # For now, we'll be conservative and not remove rows
        # Instead, we'll cap extreme values
        df_clean = df.copy()

        for col in problematic_cols:
            if method == 'iqr':
                values = df[col].dropna()
                Q1 = values.quantile(0.25)
                Q3 = values.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Cap values instead of removing rows
                df_clean[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

        logger.info(f"Capped outliers in {len(problematic_cols)} columns")
        return df_clean

    def validate_feature_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate feature distributions for ML suitability.

        Args:
            df: Dataset to validate

        Returns:
            Distribution validation report
        """
        validation_report = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            values = df[col].dropna()

            if len(values) < 10:
                continue

            # Basic statistics
            stats_report = {
                'mean': values.mean(),
                'std': values.std(),
                'min': values.min(),
                'max': values.max(),
                'skewness': values.skew(),
                'kurtosis': values.kurtosis(),
                'zeros': (values == 0).sum(),
                'negatives': (values < 0).sum()
            }

            # Distribution checks
            issues = []

            # Check for constant values
            if values.std() == 0:
                issues.append("Constant values (no variance)")

            # Check for extreme skewness
            if abs(stats_report['skewness']) > 2:
                issues.append(".2f"
            # Check for high percentage of zeros
            zero_pct = stats_report['zeros'] / len(values) * 100
            if zero_pct > 50:
                issues.append(".1f"
            # Check for negative values in non-negative features
            if stats_report['negatives'] > 0:
                neg_features = ['volume', 'liquidity', 'days_to_resolve']
                if any(f in col.lower() for f in neg_features):
                    issues.append(f"Negative values in {col}: {stats_report['negatives']} rows")

            stats_report['issues'] = issues
            validation_report[col] = stats_report

        return validation_report

    def check_class_balance(self, df: pd.DataFrame, target_col: str = 'will_resolve_yes') -> Dict[str, Any]:
        """
        Check class balance for classification tasks.

        Args:
            df: Dataset to check
            target_col: Target column name

        Returns:
            Class balance report
        """
        if target_col not in df.columns:
            return {'error': f'Target column {target_col} not found'}

        value_counts = df[target_col].value_counts()
        total = len(df)

        balance_report = {
            'class_distribution': value_counts.to_dict(),
            'total_samples': total,
            'num_classes': len(value_counts),
            'minority_class_pct': value_counts.min() / total * 100,
            'majority_class_pct': value_counts.max() / total * 100,
            'balance_ratio': value_counts.min() / value_counts.max(),
            'balanced': False
        }

        # Check if dataset is reasonably balanced
        if balance_report['balance_ratio'] > 0.1:  # At least 10% of majority class
            balance_report['balanced'] = True

        # Recommendations
        if not balance_report['balanced']:
            balance_report['recommendations'] = [
                "Consider oversampling minority class",
                "Consider undersampling majority class",
                "Use class weights in model training",
                "Consider SMOTE or other synthetic sampling"
            ]

        return balance_report

    def preprocess_for_ml(self, df: pd.DataFrame, target_col: str = 'will_resolve_yes') -> Tuple[pd.DataFrame, Dict]:
        """
        Complete preprocessing pipeline for ML training.

        Args:
            df: Raw dataset
            target_col: Target column name

        Returns:
            Tuple of (processed_dataframe, preprocessing_info)
        """
        logger.info("Starting ML preprocessing pipeline...")

        # 1. Handle missing values
        df = self.handle_missing_values(df)

        # 2. Remove/correct outliers
        df = self.remove_outliers(df)

        # 3. Encode categorical variables
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if df[col].nunique() < 20:  # Only encode low-cardinality categoricals
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self.encoders[col] = encoder

        # 4. Scale numerical features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != target_col]

        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        self.scalers['main'] = scaler

        # 5. Final validation
        quality_report = self.validate_dataset_structure(df)
        balance_report = self.check_class_balance(df, target_col)

        preprocessing_info = {
            'original_shape': (len(df), len(df.columns)),
            'quality_report': quality_report,
            'balance_report': balance_report,
            'preprocessing_steps': [
                'missing_value_imputation',
                'outlier_treatment',
                'categorical_encoding',
                'feature_scaling'
            ]
        }

        logger.info("✅ ML preprocessing completed")
        return df, preprocessing_info

    def validate_ml_readiness(self, df: pd.DataFrame, target_col: str = 'will_resolve_yes') -> Dict[str, Any]:
        """
        Comprehensive validation for ML readiness.

        Args:
            df: Dataset to validate
            target_col: Target column name

        Returns:
            Complete validation report
        """
        logger.info("Running comprehensive ML readiness validation...")

        validation_report = {
            'dataset_info': {
                'shape': df.shape,
                'memory_usage': df.memory_usage(deep=True).sum(),
                'dtypes_summary': df.dtypes.value_counts().to_dict()
            },
            'quality_check': self.validate_dataset_structure(df),
            'outlier_analysis': self.detect_outliers(df),
            'feature_distributions': self.validate_feature_distributions(df),
            'class_balance': self.check_class_balance(df, target_col),
            'recommendations': []
        }

        # Generate recommendations
        recommendations = []

        # Quality issues
        if validation_report['quality_check']['issues']:
            recommendations.extend([
                "Address data quality issues: " + issue
                for issue in validation_report['quality_check']['issues'][:3]
            ])

        # Class balance
        balance = validation_report['class_balance']
        if not balance.get('balanced', True):
            recommendations.extend(balance.get('recommendations', []))

        # Feature issues
        for col, stats in validation_report['feature_distributions'].items():
            if stats.get('issues'):
                recommendations.extend([
                    f"{col}: {issue}" for issue in stats['issues'][:2]
                ])

        validation_report['recommendations'] = list(set(recommendations))  # Remove duplicates

        # Overall readiness score
        issues_count = len(validation_report['quality_check']['issues'])
        balance_ok = validation_report['class_balance'].get('balanced', True)

        readiness_score = 100
        readiness_score -= issues_count * 10  # -10 points per issue
        if not balance_ok:
            readiness_score -= 20  # -20 for imbalanced classes

        readiness_score = max(0, min(100, readiness_score))

        validation_report['readiness_score'] = readiness_score
        validation_report['ready_for_ml'] = readiness_score >= 70

        logger.info(".1f"        return validation_report
