#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Economic Regression Analysis using Logistic Regression
Target variable: rule_violation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 1. LOAD DATA
print("=== LOADING DATA ===")
df = pd.read_csv('data/train_enhanced.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst few rows:")
print(df.head())

# 2. EXPLORE TARGET VARIABLE
print("\n=== TARGET VARIABLE ANALYSIS ===")
print("Rule violation distribution:")
print(df['rule_violation'].value_counts())
print(f"Rule violation rate: {df['rule_violation'].mean():.3f}")

# 3. DATA PREPROCESSING - CONVERT TO NUMERIC
print("\n=== DATA PREPROCESSING ===")

# Select numeric features first
numeric_features = ['body_len', 'Length_char', 'Length_word', 'NegWords', 
                   'UpperCaseRatio', 'ContainsLink', 'SentimentScore', 
                   'negScore', 'neuScore', 'posScore']

# Check which numeric features exist
available_numeric = [col for col in numeric_features if col in df.columns]
print(f"Available numeric features: {available_numeric}")

# Create feature matrix X with numeric features
X_numeric = df[available_numeric].copy()

# Handle missing values in numeric features
X_numeric = X_numeric.fillna(X_numeric.median())

# Convert categorical features to numeric
categorical_features = ['subreddit']
X_categorical = pd.DataFrame()

if 'subreddit' in df.columns:
    # Label encode subreddit
    le_subreddit = LabelEncoder()
    df['subreddit_encoded'] = le_subreddit.fit_transform(df['subreddit'].astype(str))
    X_categorical['subreddit_encoded'] = df['subreddit_encoded']
    print(f"Subreddit categories: {len(le_subreddit.classes_)}")

# Text features - simple numeric conversion
text_features = []
if 'body' in df.columns:
    # Simple text metrics
    X_categorical['body_word_count'] = df['body'].astype(str).str.split().str.len()
    X_categorical['body_char_count'] = df['body'].astype(str).str.len()
    X_categorical['body_exclamation'] = df['body'].astype(str).str.count('!')
    X_categorical['body_question'] = df['body'].astype(str).str.count(r'\?')
    X_categorical['body_uppercase_words'] = df['body'].astype(str).str.count(r'\b[A-Z]{2,}\b')
    text_features = ['body_word_count', 'body_char_count', 'body_exclamation', 
                    'body_question', 'body_uppercase_words']

# Combine all features
X = pd.concat([X_numeric, X_categorical], axis=1)
X = X.fillna(0)  # Fill any remaining NaN with 0

# Target variable
y = df['rule_violation'].astype(int)

print(f"Final feature matrix shape: {X.shape}")
print(f"Features: {list(X.columns)}")
print(f"Target variable shape: {y.shape}")

# 4. FEATURE SCALING
print("\n=== FEATURE SCALING ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Feature scaling completed")
print("Scaled features summary:")
print(X_scaled.describe())

# 5. SPLIT DATA
print("\n=== DATA SPLITTING ===")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training rule violation rate: {y_train.mean():.3f}")
print(f"Test rule violation rate: {y_test.mean():.3f}")

# 6. LOGISTIC REGRESSION MODEL
print("\n=== LOGISTIC REGRESSION MODEL ===")

# Train model
logit_model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear')
logit_model.fit(X_train, y_train)

# Predictions
y_pred = logit_model.predict(X_test)
y_pred_proba = logit_model.predict_proba(X_test)[:, 1]

print("Model training completed")

# 7. MODEL EVALUATION
print("\n=== MODEL PERFORMANCE ===")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# ROC-AUC
auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {auc_score:.4f}")

# 8. ECONOMIC INTERPRETATION
print("\n=== ECONOMIC INTERPRETATION ===")

# Coefficients and odds ratios
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logit_model.coef_[0],
    'Odds_Ratio': np.exp(logit_model.coef_[0]),
    'Abs_Coefficient': np.abs(logit_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("Feature Importance (sorted by absolute coefficient):")
print(coefficients)

# Top 10 most important features
print("\nTop 10 Most Important Features:")
top_features = coefficients.head(10)
for idx, row in top_features.iterrows():
    direction = "increases" if row['Coefficient'] > 0 else "decreases"
    print(f"{row['Feature']}: {direction} rule violation probability by {abs(row['Coefficient']):.4f}")
    print(f"  - Odds ratio: {row['Odds_Ratio']:.4f}")

# 9. FEATURE IMPORTANCE VISUALIZATION
print("\n=== CREATING VISUALIZATIONS ===")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Feature importance plot
top_10_features = coefficients.head(10)
axes[0, 0].barh(range(len(top_10_features)), top_10_features['Coefficient'])
axes[0, 0].set_yticks(range(len(top_10_features)))
axes[0, 0].set_yticklabels(top_10_features['Feature'])
axes[0, 0].set_xlabel('Coefficient')
axes[0, 0].set_title('Top 10 Feature Coefficients')
axes[0, 0].grid(True, alpha=0.3)

# 2. Odds ratios plot
axes[0, 1].barh(range(len(top_10_features)), top_10_features['Odds_Ratio'])
axes[0, 1].set_yticks(range(len(top_10_features)))
axes[0, 1].set_yticklabels(top_10_features['Feature'])
axes[0, 1].set_xlabel('Odds Ratio')
axes[0, 1].set_title('Top 10 Feature Odds Ratios')
axes[0, 1].axvline(x=1, color='red', linestyle='--', alpha=0.7)
axes[0, 1].grid(True, alpha=0.3)

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1, 0].plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.4f})')
axes[1, 0].plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.7)
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('economic_logit_analysis_results.png', dpi=300, bbox_inches='tight')
print("Visualization saved as 'economic_logit_analysis_results.png'")

# 10. STATISTICAL SUMMARY
print("\n=== STATISTICAL SUMMARY ===")
print(f"Model Accuracy: {logit_model.score(X_test, y_test):.4f}")
print(f"Intercept: {logit_model.intercept_[0]:.4f}")
print(f"Number of features: {len(X.columns)}")
print(f"Number of observations: {len(y)}")
print(f"Rule violation base rate: {y.mean():.4f}")

# Economic insights
print("\n=== ECONOMIC INSIGHTS ===")
positive_features = coefficients[coefficients['Coefficient'] > 0].head(5)
negative_features = coefficients[coefficients['Coefficient'] < 0].head(5)

print("TOP FACTORS INCREASING RULE VIOLATION PROBABILITY:")
for idx, row in positive_features.iterrows():
    print(f"- {row['Feature']}: +{row['Coefficient']:.4f} (OR: {row['Odds_Ratio']:.4f})")

print("\nTOP FACTORS DECREASING RULE VIOLATION PROBABILITY:")
for idx, row in negative_features.iterrows():
    print(f"- {row['Feature']}: {row['Coefficient']:.4f} (OR: {row['Odds_Ratio']:.4f})")

print("\n=== ANALYSIS COMPLETED ===")
print("Results saved and analysis complete!")