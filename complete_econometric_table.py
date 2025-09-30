#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Econometric Statistical Table with P-values and Descriptive Statistics
Full statistical reporting for econometric analysis
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import pearsonr, spearmanr, chi2_contingency, normaltest, jarque_bera
from scipy.stats import ttest_ind, mannwhitneyu, chi2, f_oneway, kruskal
import statsmodels.api as sm
from statsmodels.formula.api import logit
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_comprehensive_statistical_table(df, target_col='rule_violation'):
    """
    Tạo bảng thống kê kinh tế lượng hoàn chỉnh với đầy đủ p-values và thống kê mô tả
    """
    print("=" * 120)
    print("📊 COMPREHENSIVE ECONOMETRIC STATISTICAL TABLE WITH P-VALUES")
    print("=" * 120)
    
    # Initialize results list
    statistical_results = []
    
    # Separate variables by type
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_vars:
        numeric_vars.remove(target_col)
    
    categorical_vars = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"Dataset Overview:")
    print(f"  • Total observations: {len(df):,}")
    print(f"  • Numeric variables: {len(numeric_vars)}")
    print(f"  • Categorical variables: {len(categorical_vars)}")
    print(f"  • Target variable: {target_col} (Binary: {df[target_col].nunique()} levels)")
    
    # Process each variable
    all_variables = numeric_vars + categorical_vars
    
    for i, variable in enumerate(all_variables, 1):
        print(f"\nProcessing {i}/{len(all_variables)}: {variable}")
        
        var_stats = {
            'Variable': variable,
            'Variable_Type': 'Numeric' if variable in numeric_vars else 'Categorical',
            'N_Total': len(df),
            'N_Valid': df[variable].notna().sum(),
            'N_Missing': df[variable].isnull().sum(),
            'Missing_Rate_Pct': f"{(df[variable].isnull().sum() / len(df)) * 100:.2f}%"
        }
        
        if variable in numeric_vars:
            # NUMERIC VARIABLES - Complete Statistical Analysis
            valid_data = df[variable].dropna()
            
            if len(valid_data) > 0:
                # Descriptive Statistics
                var_stats.update({
                    'Mean': f"{valid_data.mean():.6f}",
                    'Std_Dev': f"{valid_data.std():.6f}",
                    'Variance': f"{valid_data.var():.6f}",
                    'Min': f"{valid_data.min():.6f}",
                    'Q1_25pct': f"{valid_data.quantile(0.25):.6f}",
                    'Median_50pct': f"{valid_data.median():.6f}",
                    'Q3_75pct': f"{valid_data.quantile(0.75):.6f}",
                    'Max': f"{valid_data.max():.6f}",
                    'Range': f"{valid_data.max() - valid_data.min():.6f}",
                    'IQR': f"{valid_data.quantile(0.75) - valid_data.quantile(0.25):.6f}",
                    'Coeff_Variation': f"{(valid_data.std() / valid_data.mean()) * 100:.4f}%" if valid_data.mean() != 0 else "N/A"
                })
                
                # Higher Moments
                skewness = stats.skew(valid_data)
                kurtosis = stats.kurtosis(valid_data)
                var_stats.update({
                    'Skewness': f"{skewness:.6f}",
                    'Kurtosis': f"{kurtosis:.6f}",
                    'Skew_Interpretation': ('Symmetric' if abs(skewness) < 0.5 else 
                                          'Moderately Skewed' if abs(skewness) < 1 else 
                                          'Highly Skewed'),
                    'Kurt_Interpretation': ('Platykurtic' if kurtosis < -1 else 
                                          'Leptokurtic' if kurtosis > 1 else 
                                          'Mesokurtic')
                })
                
                # Normality Tests with P-values
                try:
                    # Shapiro-Wilk Test (best for n < 5000)
                    if len(valid_data) <= 5000:
                        sw_stat, sw_pval = stats.shapiro(valid_data)
                        var_stats.update({
                            'Shapiro_Wilk_Stat': f"{sw_stat:.6f}",
                            'Shapiro_Wilk_P_value': f"{sw_pval:.6f}",
                            'SW_Normal': "Yes" if sw_pval > 0.05 else "No",
                            'SW_Significance': get_significance_stars(sw_pval)
                        })
                    else:
                        var_stats.update({
                            'Shapiro_Wilk_Stat': "N/A (n>5000)",
                            'Shapiro_Wilk_P_value': "N/A",
                            'SW_Normal': "N/A",
                            'SW_Significance': ""
                        })
                    
                    # Jarque-Bera Test
                    jb_stat, jb_pval = jarque_bera(valid_data)
                    var_stats.update({
                        'Jarque_Bera_Stat': f"{jb_stat:.6f}",
                        'Jarque_Bera_P_value': f"{jb_pval:.6f}",
                        'JB_Normal': "Yes" if jb_pval > 0.05 else "No",
                        'JB_Significance': get_significance_stars(jb_pval)
                    })
                    
                    # D'Agostino-Pearson Test
                    if len(valid_data) >= 8:
                        dp_stat, dp_pval = normaltest(valid_data)
                        var_stats.update({
                            'DAgostino_Pearson_Stat': f"{dp_stat:.6f}",
                            'DAgostino_Pearson_P_value': f"{dp_pval:.6f}",
                            'DP_Normal': "Yes" if dp_pval > 0.05 else "No",
                            'DP_Significance': get_significance_stars(dp_pval)
                        })
                
                except Exception as e:
                    var_stats.update({
                        'Shapiro_Wilk_Stat': "Error", 'Shapiro_Wilk_P_value': "Error",
                        'Jarque_Bera_Stat': "Error", 'Jarque_Bera_P_value': "Error",
                        'DAgostino_Pearson_Stat': "Error", 'DAgostino_Pearson_P_value': "Error"
                    })
                
                # Correlation with Target Variable
                try:
                    target_valid = df[target_col][df[variable].notna()]
                    var_valid = valid_data
                    
                    # Pearson Correlation
                    if df[target_col].nunique() == 2:  # Binary target
                        pearson_corr, pearson_pval = stats.pointbiserialr(target_valid, var_valid)
                        corr_type = "Point-Biserial"
                    else:
                        pearson_corr, pearson_pval = pearsonr(var_valid, target_valid)
                        corr_type = "Pearson"
                    
                    # Spearman Correlation
                    spearman_corr, spearman_pval = spearmanr(var_valid, target_valid)
                    
                    var_stats.update({
                        'Correlation_Type': corr_type,
                        'Pearson_Correlation': f"{pearson_corr:.6f}",
                        'Pearson_P_value': f"{pearson_pval:.6f}",
                        'Pearson_Significance': get_significance_stars(pearson_pval),
                        'Spearman_Correlation': f"{spearman_corr:.6f}",
                        'Spearman_P_value': f"{spearman_pval:.6f}",
                        'Spearman_Significance': get_significance_stars(spearman_pval)
                    })
                    
                    # Effect Size Interpretation (Cohen's conventions)
                    effect_size = abs(pearson_corr)
                    if effect_size < 0.1:
                        effect_interpret = "Negligible"
                    elif effect_size < 0.3:
                        effect_interpret = "Small"
                    elif effect_size < 0.5:
                        effect_interpret = "Medium"
                    else:
                        effect_interpret = "Large"
                    
                    var_stats['Effect_Size_Cohen'] = effect_interpret
                    
                except Exception as e:
                    var_stats.update({
                        'Correlation_Type': "Error", 'Pearson_Correlation': "Error",
                        'Pearson_P_value': "Error", 'Spearman_Correlation': "Error",
                        'Spearman_P_value': "Error", 'Effect_Size_Cohen': "Error"
                    })
                
                # Group Comparison Tests (Target = 0 vs Target = 1)
                try:
                    group_0 = df[df[target_col] == 0][variable].dropna()
                    group_1 = df[df[target_col] == 1][variable].dropna()
                    
                    if len(group_0) > 0 and len(group_1) > 0:
                        # T-test (parametric)
                        ttest_stat, ttest_pval = ttest_ind(group_0, group_1, equal_var=False)
                        
                        # Mann-Whitney U test (non-parametric)
                        mw_stat, mw_pval = mannwhitneyu(group_0, group_1, alternative='two-sided')
                        
                        # Welch's t-test effect size (Cohen's d)
                        pooled_std = np.sqrt(((len(group_0) - 1) * group_0.var() + 
                                            (len(group_1) - 1) * group_1.var()) / 
                                           (len(group_0) + len(group_1) - 2))
                        cohens_d = (group_1.mean() - group_0.mean()) / pooled_std if pooled_std != 0 else 0
                        
                        var_stats.update({
                            'Group_0_Mean': f"{group_0.mean():.6f}",
                            'Group_1_Mean': f"{group_1.mean():.6f}",
                            'Mean_Difference': f"{group_1.mean() - group_0.mean():.6f}",
                            'T_Test_Statistic': f"{ttest_stat:.6f}",
                            'T_Test_P_value': f"{ttest_pval:.6f}",
                            'T_Test_Significance': get_significance_stars(ttest_pval),
                            'Mann_Whitney_U_Stat': f"{mw_stat:.6f}",
                            'Mann_Whitney_P_value': f"{mw_pval:.6f}",
                            'MW_Significance': get_significance_stars(mw_pval),
                            'Cohens_D': f"{cohens_d:.6f}",
                            'Cohens_D_Interpretation': interpret_cohens_d(cohens_d)
                        })
                
                except Exception as e:
                    var_stats.update({
                        'Group_0_Mean': "Error", 'Group_1_Mean': "Error",
                        'T_Test_Statistic': "Error", 'T_Test_P_value': "Error",
                        'Mann_Whitney_U_Stat': "Error", 'Mann_Whitney_P_value': "Error"
                    })
                
                # Outlier Detection
                Q1 = valid_data.quantile(0.25)
                Q3 = valid_data.quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (valid_data < (Q1 - 1.5 * IQR)) | (valid_data > (Q3 + 1.5 * IQR))
                outlier_count = outlier_mask.sum()
                
                var_stats.update({
                    'Outliers_Count_IQR': f"{outlier_count}",
                    'Outliers_Percentage': f"{(outlier_count / len(valid_data)) * 100:.2f}%",
                    'Lower_Fence': f"{Q1 - 1.5 * IQR:.6f}",
                    'Upper_Fence': f"{Q3 + 1.5 * IQR:.6f}"
                })
        
        else:
            # CATEGORICAL VARIABLES - Complete Analysis
            valid_data = df[variable].dropna()
            
            if len(valid_data) > 0:
                value_counts = valid_data.value_counts()
                
                var_stats.update({
                    'Unique_Categories': f"{valid_data.nunique()}",
                    'Mode': str(value_counts.index[0]) if len(value_counts) > 0 else "N/A",
                    'Mode_Count': f"{value_counts.iloc[0]}" if len(value_counts) > 0 else "0",
                    'Mode_Frequency': f"{(value_counts.iloc[0] / len(valid_data)) * 100:.2f}%" if len(value_counts) > 0 else "0%",
                    'Second_Most_Common': str(value_counts.index[1]) if len(value_counts) > 1 else "N/A",
                    'Second_Count': f"{value_counts.iloc[1]}" if len(value_counts) > 1 else "0",
                    'Entropy': f"{stats.entropy(value_counts):.6f}",
                    'Concentration_Ratio': f"{value_counts.iloc[0] / value_counts.sum():.6f}" if len(value_counts) > 0 else "N/A"
                })
                
                # Chi-square Test of Independence
                try:
                    if valid_data.nunique() <= 50:  # Reasonable for chi-square
                        contingency_table = pd.crosstab(df[variable], df[target_col])
                        
                        if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                            chi2_stat, chi2_pval, chi2_dof, expected = chi2_contingency(contingency_table)
                            
                            # Cramer's V (effect size)
                            n = contingency_table.sum().sum()
                            cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                            
                            # Phi coefficient (for 2x2 tables)
                            if contingency_table.shape == (2, 2):
                                phi = np.sqrt(chi2_stat / n)
                                var_stats['Phi_Coefficient'] = f"{phi:.6f}"
                            
                            var_stats.update({
                                'Chi_Square_Statistic': f"{chi2_stat:.6f}",
                                'Chi_Square_P_value': f"{chi2_pval:.6f}",
                                'Chi_Square_DOF': f"{chi2_dof}",
                                'Chi_Square_Significance': get_significance_stars(chi2_pval),
                                'Cramers_V': f"{cramers_v:.6f}",
                                'Cramers_V_Interpretation': interpret_cramers_v(cramers_v),
                                'Contingency_Table_Shape': f"{contingency_table.shape[0]}x{contingency_table.shape[1]}"
                            })
                            
                            # Expected frequencies check
                            min_expected = expected.min()
                            cells_below_5 = (expected < 5).sum()
                            var_stats.update({
                                'Min_Expected_Frequency': f"{min_expected:.2f}",
                                'Cells_Below_5': f"{cells_below_5}",
                                'Chi_Square_Assumption_Met': "Yes" if min_expected >= 5 else "No"
                            })
                
                except Exception as e:
                    var_stats.update({
                        'Chi_Square_Statistic': "Error", 'Chi_Square_P_value': "Error",
                        'Cramers_V': "Error", 'Contingency_Table_Shape': "Error"
                    })
                
                # Category Distribution Analysis
                if len(value_counts) > 0:
                    # Gini coefficient for categorical distribution
                    proportions = value_counts / value_counts.sum()
                    gini = 1 - sum(proportions**2)
                    
                    var_stats.update({
                        'Gini_Coefficient': f"{gini:.6f}",
                        'Distribution_Evenness': "Even" if gini > 0.8 else "Moderate" if gini > 0.5 else "Skewed"
                    })
        
        statistical_results.append(var_stats)
    
    # Convert to DataFrame
    stats_df = pd.DataFrame(statistical_results)
    
    return stats_df

def get_significance_stars(p_value):
    """Convert p-value to significance stars"""
    try:
        p = float(p_value)
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.1:
            return "."
        else:
            return ""
    except:
        return ""

def interpret_cohens_d(d):
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "Negligible"
    elif abs_d < 0.5:
        return "Small"
    elif abs_d < 0.8:
        return "Medium"
    else:
        return "Large"

def interpret_cramers_v(v):
    """Interpret Cramer's V effect size"""
    if v < 0.1:
        return "Negligible"
    elif v < 0.3:
        return "Small"
    elif v < 0.5:
        return "Medium"
    else:
        return "Large"

def display_statistical_summary(stats_df, target_col='rule_violation'):
    """
    Display comprehensive statistical summary
    """
    print(f"\n" + "="*120)
    print("📋 COMPREHENSIVE STATISTICAL SUMMARY REPORT")
    print("="*120)
    
    # Separate by variable type
    numeric_df = stats_df[stats_df['Variable_Type'] == 'Numeric']
    categorical_df = stats_df[stats_df['Variable_Type'] == 'Categorical']
    
    print(f"\n1. NUMERIC VARIABLES DETAILED STATISTICS ({len(numeric_df)} variables)")
    print("-" * 100)
    
    if len(numeric_df) > 0:
        # Key columns for numeric variables
        numeric_display_cols = [
            'Variable', 'N_Valid', 'Missing_Rate_Pct', 'Mean', 'Std_Dev', 
            'Min', 'Max', 'Skewness', 'Kurtosis', 'Shapiro_Wilk_P_value', 'SW_Significance',
            'Jarque_Bera_P_value', 'JB_Significance', 'Pearson_Correlation', 'Pearson_P_value', 
            'Pearson_Significance', 'Effect_Size_Cohen', 'T_Test_P_value', 'T_Test_Significance',
            'Cohens_D', 'Cohens_D_Interpretation', 'Outliers_Percentage'
        ]
        
        available_numeric_cols = [col for col in numeric_display_cols if col in numeric_df.columns]
        
        # Display in chunks to avoid overwhelming output
        chunk_size = 8
        for i in range(0, len(available_numeric_cols), chunk_size):
            chunk_cols = available_numeric_cols[i:i+chunk_size]
            print(f"\nNumeric Variables - Columns {i+1}-{min(i+chunk_size, len(available_numeric_cols))}:")
            print(numeric_df[chunk_cols].to_string(index=False, max_colwidth=15))
    
    print(f"\n2. CATEGORICAL VARIABLES DETAILED STATISTICS ({len(categorical_df)} variables)")
    print("-" * 100)
    
    if len(categorical_df) > 0:
        # Key columns for categorical variables
        categorical_display_cols = [
            'Variable', 'N_Valid', 'Missing_Rate_Pct', 'Unique_Categories', 'Mode', 
            'Mode_Frequency', 'Entropy', 'Chi_Square_Statistic', 'Chi_Square_P_value',
            'Chi_Square_Significance', 'Cramers_V', 'Cramers_V_Interpretation',
            'Gini_Coefficient', 'Distribution_Evenness'
        ]
        
        available_categorical_cols = [col for col in categorical_display_cols if col in categorical_df.columns]
        
        # Display in chunks
        chunk_size = 7
        for i in range(0, len(available_categorical_cols), chunk_size):
            chunk_cols = available_categorical_cols[i:i+chunk_size]
            print(f"\nCategorical Variables - Columns {i+1}-{min(i+chunk_size, len(available_categorical_cols))}:")
            print(categorical_df[chunk_cols].to_string(index=False, max_colwidth=15))
    
    print(f"\n3. SIGNIFICANCE TESTING SUMMARY")
    print("-" * 80)
    
    # Count significant relationships
    total_vars = len(stats_df)
    significant_correlations = 0
    significant_chi_squares = 0
    significant_t_tests = 0
    
    for _, row in stats_df.iterrows():
        # Count significant correlations
        if 'Pearson_Significance' in row and row['Pearson_Significance'] in ['*', '**', '***']:
            significant_correlations += 1
        
        # Count significant chi-squares
        if 'Chi_Square_Significance' in row and row['Chi_Square_Significance'] in ['*', '**', '***']:
            significant_chi_squares += 1
        
        # Count significant t-tests
        if 'T_Test_Significance' in row and row['T_Test_Significance'] in ['*', '**', '***']:
            significant_t_tests += 1
    
    print(f"Total variables analyzed: {total_vars}")
    print(f"Significant correlations with target (p < 0.05): {significant_correlations}/{len(numeric_df)}")
    print(f"Significant chi-square tests (p < 0.05): {significant_chi_squares}/{len(categorical_df)}")
    print(f"Significant t-tests between groups (p < 0.05): {significant_t_tests}/{len(numeric_df)}")
    
    print(f"\n4. EFFECT SIZE SUMMARY")
    print("-" * 80)
    
    # Effect size distribution
    if len(numeric_df) > 0:
        effect_sizes = numeric_df['Effect_Size_Cohen'].value_counts()
        print("Effect sizes for numeric variables (Cohen's conventions):")
        for effect, count in effect_sizes.items():
            if effect != "Error":
                print(f"  {effect}: {count} variables")
    
    if len(categorical_df) > 0:
        cramers_v_sizes = categorical_df['Cramers_V_Interpretation'].value_counts()
        print("\nEffect sizes for categorical variables (Cramer's V):")
        for effect, count in cramers_v_sizes.items():
            if effect != "Error":
                print(f"  {effect}: {count} variables")
    
    print(f"\n5. DATA QUALITY ASSESSMENT")
    print("-" * 80)
    
    # Missing data summary
    total_missing = stats_df['N_Missing'].astype(str).str.replace(',', '').astype(float).sum()
    high_missing_vars = len(stats_df[stats_df['Missing_Rate_Pct'].str.replace('%', '').astype(float) > 10])
    
    print(f"Total missing values across all variables: {total_missing:,.0f}")
    print(f"Variables with >10% missing data: {high_missing_vars}")
    
    # Outlier summary for numeric variables
    if len(numeric_df) > 0:
        high_outlier_vars = len(numeric_df[numeric_df['Outliers_Percentage'].str.replace('%', '').astype(float) > 5])
        print(f"Numeric variables with >5% outliers: {high_outlier_vars}")
    
    # Normality summary
    if len(numeric_df) > 0:
        normal_vars_sw = len(numeric_df[numeric_df['SW_Normal'] == 'Yes'])
        normal_vars_jb = len(numeric_df[numeric_df['JB_Normal'] == 'Yes'])
        print(f"Variables passing Shapiro-Wilk normality test: {normal_vars_sw}/{len(numeric_df)}")
        print(f"Variables passing Jarque-Bera normality test: {normal_vars_jb}/{len(numeric_df)}")

if __name__ == "__main__":
    print("🔬 Starting Comprehensive Econometric Statistical Analysis...")
    
    # Load data
    df = pd.read_csv('data/train_enhanced.csv')
    print(f"Data loaded: {df.shape}")
    
    # Create comprehensive statistical table
    stats_table = create_comprehensive_statistical_table(df, target_col='rule_violation')
    
    # Display summary
    display_statistical_summary(stats_table, target_col='rule_violation')
    
    # Save complete table
    stats_table.to_csv('comprehensive_econometric_statistics.csv', index=False)
    
    # Save separate tables for numeric and categorical
    numeric_stats = stats_table[stats_table['Variable_Type'] == 'Numeric']
    categorical_stats = stats_table[stats_table['Variable_Type'] == 'Categorical']
    
    numeric_stats.to_csv('numeric_variables_statistics.csv', index=False)
    categorical_stats.to_csv('categorical_variables_statistics.csv', index=False)
    
    print(f"\n" + "="*120)
    print("✅ COMPREHENSIVE ECONOMETRIC ANALYSIS COMPLETED!")
    print("="*120)
    print("\nFiles created:")
    print("📊 comprehensive_econometric_statistics.csv - Complete statistical table")
    print("📈 numeric_variables_statistics.csv - Numeric variables only")
    print("📋 categorical_variables_statistics.csv - Categorical variables only")
    print("\n📊 Statistical tests included:")
    print("• Descriptive statistics (mean, std, quartiles, skewness, kurtosis)")
    print("• Normality tests (Shapiro-Wilk, Jarque-Bera, D'Agostino-Pearson)")
    print("• Correlation analysis (Pearson, Spearman) with p-values")
    print("• Group comparison tests (t-test, Mann-Whitney U)")
    print("• Chi-square tests of independence with Cramer's V")
    print("• Effect size measures (Cohen's d, Cramer's V, Cohen's conventions)")
    print("• Outlier detection (IQR method)")
    print("• Missing data analysis")
    print("• Statistical significance testing with p-values")
    print("• Comprehensive interpretation and reporting")