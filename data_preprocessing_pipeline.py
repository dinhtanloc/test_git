#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data Preprocessing Pipeline for Economic Analysis
Complete preprocessing and statistical table generation
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import warnings
warnings.filterwarnings('ignore')

def comprehensive_data_preprocessing(df):
    """
    Tiền xử lý dữ liệu hoàn chỉnh
    """
    print("=" * 100)
    print("🔧 COMPREHENSIVE DATA PREPROCESSING PIPELINE")
    print("=" * 100)
    
    df_processed = df.copy()
    processing_log = []
    
    # 1. BASIC INFO
    print(f"\n1. ORIGINAL DATA OVERVIEW")
    print("-" * 60)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"Data types:\n{df.dtypes.value_counts()}")
    
    processing_log.append({
        'Step': 'Original Data',
        'Shape': df.shape,
        'Numeric_Cols': len(df.select_dtypes(include=[np.number]).columns),
        'Text_Cols': len(df.select_dtypes(include=['object']).columns),
        'Missing_Values': df.isnull().sum().sum()
    })
    
    # 2. HANDLE MISSING VALUES
    print(f"\n2. MISSING VALUES HANDLING")
    print("-" * 60)
    
    missing_summary = df.isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    
    if len(missing_cols) > 0:
        print("Missing values found:")
        for col, count in missing_cols.items():
            pct = (count / len(df)) * 100
            print(f"  {col}: {count} ({pct:.2f}%)")
            
            if df[col].dtype in ['object']:
                df_processed[col] = df_processed[col].fillna('missing_value')
            else:
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    else:
        print("No missing values found! ✅")
    
    processing_log.append({
        'Step': 'Missing Values Handled',
        'Shape': df_processed.shape,
        'Missing_Values': df_processed.isnull().sum().sum()
    })
    
    # 3. TEXT COLUMNS IDENTIFICATION AND PROCESSING
    print(f"\n3. TEXT COLUMNS IDENTIFICATION & PROCESSING")
    print("-" * 60)
    
    text_columns = []
    categorical_columns = []
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target from numeric if present
    if 'rule_violation' in numeric_columns:
        numeric_columns.remove('rule_violation')
    
    # Identify text vs categorical columns
    for col in df_processed.select_dtypes(include=['object']).columns:
        if col == 'rule_violation':
            continue
            
        unique_ratio = df_processed[col].nunique() / len(df_processed)
        avg_length = df_processed[col].astype(str).str.len().mean()
        
        if unique_ratio > 0.1 and avg_length > 15:  # Text columns
            text_columns.append(col)
        else:  # Categorical columns
            categorical_columns.append(col)
    
    print(f"Text columns identified ({len(text_columns)}): {text_columns}")
    print(f"Categorical columns ({len(categorical_columns)}): {categorical_columns}")
    print(f"Numeric columns ({len(numeric_columns)}): {numeric_columns}")
    
    # 4. TEXT PREPROCESSING
    if text_columns:
        print(f"\n4. TEXT PREPROCESSING")
        print("-" * 60)
        
        for col in text_columns:
            print(f"Processing text column: {col}")
            
            # Basic text cleaning
            df_processed[f'{col}_original'] = df_processed[col].copy()  # Keep original
            
            # Clean text
            df_processed[col] = df_processed[col].astype(str)
            df_processed[col] = df_processed[col].str.lower()
            df_processed[col] = df_processed[col].str.replace(r'http\S+|www\S+', 'URL', regex=True)
            df_processed[col] = df_processed[col].str.replace(r'[^\w\s]', ' ', regex=True)
            df_processed[col] = df_processed[col].str.replace(r'\s+', ' ', regex=True)
            df_processed[col] = df_processed[col].str.strip()
            
            # Text features extraction
            df_processed[f'{col}_length'] = df_processed[f'{col}_original'].astype(str).str.len()
            df_processed[f'{col}_word_count'] = df_processed[f'{col}_original'].astype(str).str.split().str.len()
            df_processed[f'{col}_unique_words'] = df_processed[f'{col}_original'].astype(str).apply(
                lambda x: len(set(str(x).split())) if pd.notna(x) else 0
            )
            df_processed[f'{col}_uppercase_ratio'] = df_processed[f'{col}_original'].astype(str).apply(
                lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
            )
            df_processed[f'{col}_exclamation_count'] = df_processed[f'{col}_original'].astype(str).str.count('!')
            df_processed[f'{col}_question_count'] = df_processed[f'{col}_original'].astype(str).str.count(r'\?')
            df_processed[f'{col}_url_count'] = df_processed[f'{col}_original'].astype(str).str.count(r'http|www')
            
            print(f"  - Created 7 features from {col}")
    
    # 5. CATEGORICAL ENCODING
    if categorical_columns:
        print(f"\n5. CATEGORICAL ENCODING")
        print("-" * 60)
        
        label_encoders = {}
        
        for col in categorical_columns:
            print(f"Encoding categorical column: {col}")
            
            # Keep original
            df_processed[f'{col}_original'] = df_processed[col].copy()
            
            # Label encoding
            le = LabelEncoder()
            df_processed[f'{col}_encoded'] = le.fit_transform(df_processed[col].astype(str))
            
            # One-hot encoding for columns with few categories (≤5)
            if df_processed[col].nunique() <= 5:
                dummies = pd.get_dummies(df_processed[col], prefix=f'{col}_dummy')
                df_processed = pd.concat([df_processed, dummies], axis=1)
                print(f"  - Label encoded + One-hot encoded ({df_processed[col].nunique()} categories)")
            else:
                print(f"  - Label encoded only ({df_processed[col].nunique()} categories)")
            
            label_encoders[col] = le
    
    # 6. NUMERIC FEATURES ENGINEERING
    print(f"\n6. NUMERIC FEATURES ENGINEERING")
    print("-" * 60)
    
    if numeric_columns:
        for col in numeric_columns:
            print(f"Engineering features for: {col}")
            
            # Original values
            df_processed[f'{col}_original'] = df_processed[col].copy()
            
            # Log transformation for skewed data
            if df_processed[col].min() > 0:  # Only if all values are positive
                df_processed[f'{col}_log'] = np.log1p(df_processed[col])
            
            # Square root transformation
            if df_processed[col].min() >= 0:  # Only if all values are non-negative
                df_processed[f'{col}_sqrt'] = np.sqrt(df_processed[col])
            
            # Z-score (standardized)
            df_processed[f'{col}_zscore'] = (df_processed[col] - df_processed[col].mean()) / df_processed[col].std()
            
            # Min-max scaled (0-1)
            scaler = MinMaxScaler()
            df_processed[f'{col}_minmax'] = scaler.fit_transform(df_processed[[col]]).flatten()
            
            # Binning (quartiles)
            try:
                df_processed[f'{col}_quartile'] = pd.qcut(df_processed[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'], duplicates='drop')
            except ValueError:
                # If qcut fails, use cut instead
                df_processed[f'{col}_quartile'] = pd.cut(df_processed[col], bins=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            
            print(f"  - Created 5 engineered features from {col}")
    
    # 7. OUTLIER TREATMENT
    print(f"\n7. OUTLIER DETECTION & TREATMENT")
    print("-" * 60)
    
    outlier_summary = {}
    
    for col in numeric_columns:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((df_processed[col] < lower_bound) | (df_processed[col] > upper_bound)).sum()
        outlier_pct = (outliers / len(df_processed)) * 100
        
        outlier_summary[col] = {'count': outliers, 'percentage': outlier_pct}
        
        # Create outlier-treated version (winsorizing)
        df_processed[f'{col}_winsorized'] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
        
        if outlier_pct > 1:
            print(f"  {col}: {outliers} outliers ({outlier_pct:.2f}%) - Winsorized")
    
    # 8. FINAL PROCESSING SUMMARY
    print(f"\n8. PROCESSING SUMMARY")
    print("-" * 60)
    
    final_shape = df_processed.shape
    feature_increase = final_shape[1] - df.shape[1]
    
    print(f"Original shape: {df.shape}")
    print(f"Final shape: {final_shape}")
    print(f"Features added: {feature_increase}")
    print(f"Memory usage: {df_processed.memory_usage().sum() / 1024**2:.2f} MB")
    
    processing_log.append({
        'Step': 'Final Processed',
        'Shape': final_shape,
        'Features_Added': feature_increase,
        'Text_Columns_Processed': len(text_columns),
        'Categorical_Encoded': len(categorical_columns),
        'Numeric_Engineered': len(numeric_columns)
    })
    
    return df_processed, processing_log, outlier_summary

def create_processed_statistics_table(df_processed):
    """
    Tạo bảng thống kê cho dữ liệu đã xử lý
    """
    print(f"\n" + "=" * 100)
    print("📊 PROCESSED DATA STATISTICS TABLE")
    print("=" * 100)
    
    stats_summary = []
    
    # Get all numeric columns (including engineered features)
    numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    if 'rule_violation' in numeric_cols:
        numeric_cols.remove('rule_violation')
    
    # Get categorical columns (including encoded)
    categorical_cols = df_processed.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"Analyzing {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns...")
    
    # Analyze numeric columns
    for col in numeric_cols[:20]:  # Limit to first 20 for display
        col_stats = {
            'Column': col,
            'Type': 'Numeric',
            'Count': df_processed[col].count(),
            'Missing': df_processed[col].isnull().sum(),
            'Mean': f"{df_processed[col].mean():.4f}" if pd.notna(df_processed[col].mean()) else 'NaN',
            'Std': f"{df_processed[col].std():.4f}" if pd.notna(df_processed[col].std()) else 'NaN',
            'Min': f"{df_processed[col].min():.4f}" if pd.notna(df_processed[col].min()) else 'NaN',
            'Max': f"{df_processed[col].max():.4f}" if pd.notna(df_processed[col].max()) else 'NaN',
            'Skewness': f"{stats.skew(df_processed[col].dropna()):.4f}" if len(df_processed[col].dropna()) > 0 else 'NaN',
            'Unique_Values': df_processed[col].nunique()
        }
        
        # Correlation with target if available
        if 'rule_violation' in df_processed.columns:
            try:
                corr, p_val = stats.pearsonr(df_processed[col].dropna(), 
                                           df_processed['rule_violation'][df_processed[col].notna()])
                col_stats['Correlation_Target'] = f"{corr:.4f}"
                col_stats['P_Value'] = f"{p_val:.4f}"
                col_stats['Significance'] = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            except:
                col_stats['Correlation_Target'] = 'Error'
                col_stats['P_Value'] = 'Error'
                col_stats['Significance'] = ''
        
        stats_summary.append(col_stats)
    
    # Analyze categorical columns
    for col in categorical_cols[:10]:  # Limit to first 10 for display
        col_stats = {
            'Column': col,
            'Type': 'Categorical',
            'Count': df_processed[col].count(),
            'Missing': df_processed[col].isnull().sum(),
            'Unique_Values': df_processed[col].nunique(),
            'Most_Frequent': str(df_processed[col].mode().iloc[0]) if len(df_processed[col].mode()) > 0 else 'N/A',
            'Most_Freq_Count': df_processed[col].value_counts().iloc[0] if len(df_processed[col].value_counts()) > 0 else 0,
            'Entropy': f"{stats.entropy(df_processed[col].value_counts()):.4f}" if df_processed[col].nunique() > 1 else '0.0000'
        }
        stats_summary.append(col_stats)
    
    # Convert to DataFrame and display
    stats_df = pd.DataFrame(stats_summary)
    
    # Display by type
    numeric_stats = stats_df[stats_df['Type'] == 'Numeric']
    categorical_stats = stats_df[stats_df['Type'] == 'Categorical']
    
    if len(numeric_stats) > 0:
        print(f"\nNUMERIC VARIABLES STATISTICS (showing first 20):")
        print("-" * 80)
        display_cols = ['Column', 'Count', 'Missing', 'Mean', 'Std', 'Min', 'Max', 
                       'Skewness', 'Correlation_Target', 'P_Value', 'Significance']
        available_cols = [col for col in display_cols if col in numeric_stats.columns]
        print(numeric_stats[available_cols].to_string(index=False, max_colwidth=20))
    
    if len(categorical_stats) > 0:
        print(f"\nCATEGORICAL VARIABLES STATISTICS (showing first 10):")
        print("-" * 80)
        display_cols = ['Column', 'Count', 'Missing', 'Unique_Values', 'Most_Frequent', 
                       'Most_Freq_Count', 'Entropy']
        available_cols = [col for col in display_cols if col in categorical_stats.columns]
        print(categorical_stats[available_cols].to_string(index=False, max_colwidth=20))
    
    return stats_df

if __name__ == "__main__":
    print("🚀 Starting Comprehensive Data Preprocessing...")
    
    # Load original data
    print("Loading original data...")
    df_original = pd.read_csv('data/train_enhanced.csv')
    print(f"Original data loaded: {df_original.shape}")
    
    # Run comprehensive preprocessing
    df_processed, processing_log, outlier_summary = comprehensive_data_preprocessing(df_original)
    
    # Save processed data
    output_file = 'train_df_processing.csv'
    df_processed.to_csv(output_file, index=False)
    print(f"\n✅ Processed data saved to: {output_file}")
    
    # Create and display statistics table
    stats_df = create_processed_statistics_table(df_processed)
    
    # Save statistics table
    stats_df.to_csv('processed_data_statistics.csv', index=False)
    print(f"\n✅ Statistics table saved to: processed_data_statistics.csv")
    
    # Display processing summary
    print(f"\n" + "=" * 100)
    print("📋 PROCESSING PIPELINE SUMMARY")
    print("=" * 100)
    
    processing_df = pd.DataFrame(processing_log)
    print(processing_df.to_string(index=False))
    
    print(f"\n🎯 KEY IMPROVEMENTS:")
    print(f"• Original features: {df_original.shape[1]}")
    print(f"• Final features: {df_processed.shape[1]}")
    print(f"• Features added: {df_processed.shape[1] - df_original.shape[1]}")
    print(f"• Text columns processed: {len([col for col in df_processed.columns if '_length' in col])}")
    print(f"• Categorical columns encoded: {len([col for col in df_processed.columns if '_encoded' in col])}")
    print(f"• Numeric features engineered: {len([col for col in df_processed.columns if '_log' in col or '_sqrt' in col])}")
    
    print(f"\n📁 FILES CREATED:")
    print(f"📊 train_df_processing.csv - Fully processed dataset")
    print(f"📈 processed_data_statistics.csv - Statistical summary table")
    
    print(f"\n🔬 READY FOR ECONOMETRIC ANALYSIS!")
    print(f"Use 'train_df_processing.csv' for your econometric models")