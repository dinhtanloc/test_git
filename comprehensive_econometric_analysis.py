#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Econometric Statistical Analysis using Scipy
Full dataset statistical table analysis - All columns
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency, normaltest, jarque_bera
from scipy.stats import ttest_ind, mannwhitneyu, chi2, f_oneway
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def comprehensive_descriptive_stats(df, target_col='rule_violation'):
    """
    Tạo bảng thống kê mô tả hoàn chỉnh cho tất cả các cột
    """
    print("=" * 100)
    print("📊 COMPREHENSIVE ECONOMETRIC STATISTICAL ANALYSIS")
    print("=" * 100)
    
    # 1. DATASET OVERVIEW
    print(f"\n1. DATASET OVERVIEW")
    print("-" * 60)
    print(f"Dataset shape: {df.shape}")
    print(f"Total observations: {df.shape[0]:,}")
    print(f"Total variables: {df.shape[1]}")
    print(f"Target variable: {target_col}")
    print(f"Missing values: {df.isnull().sum().sum()}")
    
    # 2. TARGET VARIABLE DETAILED ANALYSIS
    print(f"\n2. TARGET VARIABLE ANALYSIS: {target_col}")
    print("-" * 60)
    target_stats = df[target_col].describe()
    print(target_stats)
    
    if df[target_col].nunique() <= 10:
        print(f"\nTarget distribution:")
        target_dist = df[target_col].value_counts().sort_index()
        target_prop = df[target_col].value_counts(normalize=True).sort_index()
        for val in target_dist.index:
            print(f"  {val}: {target_dist[val]:,} ({target_prop[val]:.1%})")
    
    # 3. COMPREHENSIVE STATISTICAL TABLE
    print(f"\n3. COMPREHENSIVE STATISTICAL SUMMARY TABLE")
    print("=" * 140)
    
    # Create comprehensive statistics table
    stats_results = []
    
    for col in df.columns:
        if col == target_col:
            continue
            
        print(f"\nProcessing: {col}")
        
        col_data = {
            'Variable': col,
            'Data_Type': str(df[col].dtype),
            'N_Valid': df[col].notna().sum(),
            'N_Missing': df[col].isnull().sum(),
            'Missing_Rate': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}%",
            'Unique_Values': df[col].nunique()
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            # NUMERIC VARIABLES - Full Statistical Analysis
            valid_data = df[col].dropna()
            
            if len(valid_data) > 0:
                # Basic descriptive statistics
                col_data.update({
                    'Mean': f"{valid_data.mean():.4f}",
                    'Std_Dev': f"{valid_data.std():.4f}",
                    'Min': f"{valid_data.min():.4f}",
                    'Q1_25%': f"{valid_data.quantile(0.25):.4f}",
                    'Median_50%': f"{valid_data.median():.4f}",
                    'Q3_75%': f"{valid_data.quantile(0.75):.4f}",
                    'Max': f"{valid_data.max():.4f}",
                    'Range': f"{valid_data.max() - valid_data.min():.4f}",
                    'IQR': f"{valid_data.quantile(0.75) - valid_data.quantile(0.25):.4f}",
                    'CV': f"{(valid_data.std() / valid_data.mean()) * 100:.2f}%" if valid_data.mean() != 0 else "N/A"
                })
                
                # Advanced statistical measures
                col_data.update({
                    'Skewness': f"{stats.skew(valid_data):.4f}",
                    'Kurtosis': f"{stats.kurtosis(valid_data):.4f}",
                    'Variance': f"{valid_data.var():.4f}"
                })
                
                # Distribution shape analysis
                skew_val = stats.skew(valid_data)
                kurt_val = stats.kurtosis(valid_data)
                
                if abs(skew_val) < 0.5:
                    skew_interpret = "Symmetric"
                elif abs(skew_val) < 1:
                    skew_interpret = "Moderately Skewed"
                else:
                    skew_interpret = "Highly Skewed"
                
                if kurt_val < -1:
                    kurt_interpret = "Platykurtic"
                elif kurt_val > 1:
                    kurt_interpret = "Leptokurtic"
                else:
                    kurt_interpret = "Mesokurtic"
                
                col_data.update({
                    'Skew_Interpretation': skew_interpret,
                    'Kurt_Interpretation': kurt_interpret
                })
                
                # Normality tests
                try:
                    if len(valid_data) >= 8:
                        # Jarque-Bera test
                        jb_stat, jb_pval = jarque_bera(valid_data)
                        col_data.update({
                            'JB_Statistic': f"{jb_stat:.4f}",
                            'JB_P_value': f"{jb_pval:.4f}",
                            'Normal_JB': "Yes" if jb_pval > 0.05 else "No"
                        })
                        
                        # Shapiro-Wilk test (if sample size allows)
                        if len(valid_data) <= 5000:
                            sw_stat, sw_pval = stats.shapiro(valid_data)
                            col_data.update({
                                'SW_Statistic': f"{sw_stat:.4f}",
                                'SW_P_value': f"{sw_pval:.4f}",
                                'Normal_SW': "Yes" if sw_pval > 0.05 else "No"
                            })
                except Exception as e:
                    col_data.update({
                        'JB_Statistic': "N/A", 'JB_P_value': "N/A", 'Normal_JB': "N/A",
                        'SW_Statistic': "N/A", 'SW_P_value': "N/A", 'Normal_SW': "N/A"
                    })
                
                # Correlation with target variable
                try:
                    target_valid = df[target_col][df[col].notna()]
                    
                    if df[target_col].nunique() == 2:  # Binary target
                        corr_coef, corr_pval = stats.pointbiserialr(target_valid, valid_data)
                        corr_type = "Point-Biserial"
                    else:
                        corr_coef, corr_pval = pearsonr(valid_data, target_valid)
                        corr_type = "Pearson"
                    
                    # Spearman correlation
                    spear_coef, spear_pval = spearmanr(valid_data, target_valid)
                    
                    # Significance stars
                    corr_sig = "***" if corr_pval < 0.001 else "**" if corr_pval < 0.01 else "*" if corr_pval < 0.05 else ""
                    spear_sig = "***" if spear_pval < 0.001 else "**" if spear_pval < 0.01 else "*" if spear_pval < 0.05 else ""
                    
                    col_data.update({
                        'Corr_Type': corr_type,
                        'Pearson_Corr': f"{corr_coef:.4f}{corr_sig}",
                        'Pearson_P_val': f"{corr_pval:.4f}",
                        'Spearman_Corr': f"{spear_coef:.4f}{spear_sig}",
                        'Spearman_P_val': f"{spear_pval:.4f}"
                    })
                    
                    # Effect size interpretation
                    effect_size = abs(corr_coef)
                    if effect_size < 0.1:
                        effect_interpret = "Negligible"
                    elif effect_size < 0.3:
                        effect_interpret = "Small"
                    elif effect_size < 0.5:
                        effect_interpret = "Medium" 
                    else:
                        effect_interpret = "Large"
                    
                    col_data['Effect_Size'] = effect_interpret
                    
                except Exception as e:
                    col_data.update({
                        'Corr_Type': "N/A", 'Pearson_Corr': "N/A", 'Pearson_P_val': "N/A",
                        'Spearman_Corr': "N/A", 'Spearman_P_val': "N/A", 'Effect_Size': "N/A"
                    })
                
                # Outlier detection using IQR method
                Q1 = valid_data.quantile(0.25)
                Q3 = valid_data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_count = ((valid_data < (Q1 - 1.5 * IQR)) | (valid_data > (Q3 + 1.5 * IQR))).sum()
                outlier_pct = (outlier_count / len(valid_data)) * 100
                
                col_data.update({
                    'Outliers_Count': f"{outlier_count}",
                    'Outliers_Pct': f"{outlier_pct:.2f}%"
                })
        
        else:
            # CATEGORICAL VARIABLES - Full Analysis
            valid_data = df[col].dropna()
            
            if len(valid_data) > 0:
                value_counts = valid_data.value_counts()
                
                col_data.update({
                    'Mode': str(value_counts.index[0]) if len(value_counts) > 0 else "N/A",
                    'Mode_Count': f"{value_counts.iloc[0]}" if len(value_counts) > 0 else "0",
                    'Mode_Frequency': f"{(value_counts.iloc[0] / len(valid_data)) * 100:.2f}%" if len(value_counts) > 0 else "0%",
                    'Entropy': f"{stats.entropy(value_counts):.4f}"
                })
                
                # Chi-square test with target (if applicable)
                try:
                    if df[target_col].nunique() <= 10 and valid_data.nunique() <= 50:  # Reasonable for chi-square
                        contingency_table = pd.crosstab(df[col], df[target_col])
                        chi2_stat, chi2_pval, dof, expected = chi2_contingency(contingency_table)
                        
                        # Cramer's V for effect size
                        n = contingency_table.sum().sum()
                        cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                        
                        chi2_sig = "***" if chi2_pval < 0.001 else "**" if chi2_pval < 0.01 else "*" if chi2_pval < 0.05 else ""
                        
                        col_data.update({
                            'Chi2_Statistic': f"{chi2_stat:.4f}{chi2_sig}",
                            'Chi2_P_value': f"{chi2_pval:.4f}",
                            'Chi2_DOF': f"{dof}",
                            'Cramers_V': f"{cramers_v:.4f}"
                        })
                        
                        # Effect size interpretation for Cramer's V
                        if cramers_v < 0.1:
                            effect_interpret = "Negligible"
                        elif cramers_v < 0.3:
                            effect_interpret = "Small"
                        elif cramers_v < 0.5:
                            effect_interpret = "Medium"
                        else:
                            effect_interpret = "Large"
                        
                        col_data['Effect_Size'] = effect_interpret
                        
                except Exception as e:
                    col_data.update({
                        'Chi2_Statistic': "N/A", 'Chi2_P_value': "N/A", 
                        'Chi2_DOF': "N/A", 'Cramers_V': "N/A", 'Effect_Size': "N/A"
                    })
        
        stats_results.append(col_data)
    
    # Convert to DataFrame for better display
    stats_df = pd.DataFrame(stats_results)
    
    return stats_df

def create_econometric_report(df, target_col='rule_violation'):
    """
    Tạo báo cáo kinh tế lượng hoàn chỉnh
    """
    
    # Get comprehensive statistics
    stats_table = comprehensive_descriptive_stats(df, target_col)
    
    # Display results in sections
    print(f"\n4. DETAILED STATISTICAL RESULTS BY VARIABLE TYPE")
    print("=" * 100)
    
    # Separate numeric and categorical variables
    numeric_vars = stats_table[stats_table['Data_Type'].str.contains('int|float')]
    categorical_vars = stats_table[~stats_table['Data_Type'].str.contains('int|float')]
    
    print(f"\n4.1 NUMERIC VARIABLES SUMMARY ({len(numeric_vars)} variables)")
    print("-" * 80)
    
    if len(numeric_vars) > 0:
        # Key numeric statistics
        numeric_display_cols = ['Variable', 'N_Valid', 'Missing_Rate', 'Mean', 'Std_Dev', 
                               'Min', 'Max', 'Skewness', 'Kurtosis', 'Normal_JB', 
                               'Pearson_Corr', 'Effect_Size', 'Outliers_Pct']
        
        available_cols = [col for col in numeric_display_cols if col in numeric_vars.columns]
        print(numeric_vars[available_cols].to_string(index=False))
    
    print(f"\n4.2 CATEGORICAL VARIABLES SUMMARY ({len(categorical_vars)} variables)")
    print("-" * 80)
    
    if len(categorical_vars) > 0:
        # Key categorical statistics
        categorical_display_cols = ['Variable', 'N_Valid', 'Missing_Rate', 'Unique_Values', 
                                   'Mode', 'Mode_Frequency', 'Entropy', 'Chi2_Statistic', 'Effect_Size']
        
        available_cols = [col for col in categorical_display_cols if col in categorical_vars.columns]
        print(categorical_vars[available_cols].to_string(index=False))
    
    # 5. CORRELATION MATRIX FOR NUMERIC VARIABLES
    print(f"\n5. CORRELATION ANALYSIS")
    print("-" * 60)
    
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_columns) > 1:
        corr_matrix = df[numeric_columns].corr()
        
        print("Top 10 Strongest Correlations with Target Variable:")
        target_corrs = corr_matrix[target_col].abs().sort_values(ascending=False)
        for var, corr in target_corrs.head(10).items():
            if var != target_col:
                direction = "+" if corr_matrix[target_col][var] > 0 else "-"
                print(f"  {var}: {direction}{corr:.4f}")
    
    # 6. STATISTICAL SIGNIFICANCE SUMMARY
    print(f"\n6. STATISTICAL SIGNIFICANCE SUMMARY")
    print("-" * 60)
    
    # Count significant relationships
    significant_vars = []
    
    for _, row in stats_table.iterrows():
        if 'Pearson_P_val' in row and pd.notna(row['Pearson_P_val']):
            try:
                p_val = float(row['Pearson_P_val'])
                if p_val < 0.05:
                    significant_vars.append((row['Variable'], p_val, 'Pearson'))
            except:
                pass
        
        if 'Chi2_P_value' in row and pd.notna(row['Chi2_P_value']):
            try:
                p_val = float(row['Chi2_P_value'])
                if p_val < 0.05:
                    significant_vars.append((row['Variable'], p_val, 'Chi-square'))
            except:
                pass
    
    print(f"Variables with significant relationship to target (p < 0.05): {len(significant_vars)}")
    for var, p_val, test_type in sorted(significant_vars, key=lambda x: x[1]):
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
        print(f"  {var}: p = {p_val:.4f} {significance} ({test_type})")
    
    # Save complete table to CSV
    stats_table.to_csv('comprehensive_econometric_statistics.csv', index=False)
    print(f"\n✅ Complete statistical table saved to 'comprehensive_econometric_statistics.csv'")
    
    return stats_table

if __name__ == "__main__":
    print("🔬 Loading data for comprehensive econometric analysis...")
    
    # Load data
    df = pd.read_csv('data/train_enhanced.csv')
    
    print(f"Data loaded: {df.shape[0]} observations, {df.shape[1]} variables")
    
    # Run comprehensive analysis
    results = create_econometric_report(df, target_col='rule_violation')
    
    print("\n" + "="*100)
    print("📊 ECONOMETRIC ANALYSIS COMPLETED!")
    print("="*100)
    print("\nFiles created:")
    print("• comprehensive_econometric_statistics.csv - Complete statistical table")
    print("\nAnalysis includes:")
    print("• Descriptive statistics for all variables")
    print("• Normality tests (Jarque-Bera, Shapiro-Wilk)")
    print("• Correlation analysis (Pearson, Spearman)")
    print("• Chi-square tests for categorical variables")
    print("• Effect size measures (Cohen's conventions)")
    print("• Outlier detection and missing value analysis")
    print("• Statistical significance testing")