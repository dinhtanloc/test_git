#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Economic Regression Analysis with Better Text Processing
Target variable: rule_violation
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def advanced_text_features(text_series):
    """
    Tạo các features nâng cao từ text
    """
    features = pd.DataFrame()
    
    # Chuyển đổi thành string và xử lý NaN
    text_clean = text_series.astype(str).fillna('')
    
    # 1. Basic counting features
    features['word_count'] = text_clean.str.split().str.len()
    features['char_count'] = text_clean.str.len()
    features['sentence_count'] = text_clean.str.count(r'[.!?]+')
    features['avg_word_length'] = text_clean.apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)
    
    # 2. Punctuation features
    features['exclamation_count'] = text_clean.str.count('!')
    features['question_count'] = text_clean.str.count(r'\?')
    features['comma_count'] = text_clean.str.count(',')
    features['period_count'] = text_clean.str.count(r'\.')
    features['punctuation_ratio'] = text_clean.str.count(r'[^\w\s]') / features['char_count'].replace(0, 1)
    
    # 3. Case features  
    features['uppercase_words'] = text_clean.str.count(r'\b[A-Z]{2,}\b')
    features['uppercase_ratio'] = text_clean.apply(lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0)
    features['title_words'] = text_clean.str.count(r'\b[A-Z][a-z]+\b')
    
    # 4. Special characters and patterns
    features['digit_count'] = text_clean.str.count(r'\d')
    features['url_count'] = text_clean.str.count(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    features['email_count'] = text_clean.str.count(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    features['hashtag_count'] = text_clean.str.count(r'#\w+')
    features['mention_count'] = text_clean.str.count(r'@\w+')
    
    # 5. Suspicious patterns (for rule violation detection)
    spam_keywords = ['click here', 'buy now', 'free', 'win', 'money', 'offer', 'deal', 'discount', 'urgent']
    features['spam_keywords'] = text_clean.apply(lambda x: sum(1 for keyword in spam_keywords if keyword.lower() in x.lower()))
    
    # 6. Emotional indicators
    positive_words = ['good', 'great', 'awesome', 'excellent', 'love', 'like', 'best', 'amazing', 'perfect']
    negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'sucks', 'stupid', 'disgusting']
    
    features['positive_words'] = text_clean.apply(lambda x: sum(1 for word in positive_words if word.lower() in x.lower()))
    features['negative_words'] = text_clean.apply(lambda x: sum(1 for word in negative_words if word.lower() in x.lower()))
    
    # 7. Text quality indicators
    features['repeated_chars'] = text_clean.str.count(r'(.)\1{2,}')  # aaaaaa, !!!!!
    features['caps_lock_words'] = text_clean.str.count(r'\b[A-Z]{3,}\b')
    features['whitespace_ratio'] = text_clean.str.count(r'\s') / features['char_count'].replace(0, 1)
    
    # Fill NaN values
    features = features.fillna(0)
    
    return features

def create_tfidf_features(text_series, max_features=100):
    """
    Tạo TF-IDF features từ text
    """
    # Xử lý text
    text_clean = text_series.astype(str).fillna('')
    
    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(
        max_features=max_features,
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=2,  # ignore terms that appear in less than 2 documents
        max_df=0.95  # ignore terms that appear in more than 95% of documents
    )
    
    try:
        tfidf_matrix = tfidf.fit_transform(text_clean)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f'tfidf_{feature}' for feature in tfidf.get_feature_names_out()]
        )
        return tfidf_df, tfidf
    except:
        # Fallback if TF-IDF fails
        return pd.DataFrame(), None

# 1. LOAD DATA
print("=== LOADING DATA ===")
df = pd.read_csv('data/train_enhanced.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# 2. EXPLORE TARGET VARIABLE
print("\n=== TARGET VARIABLE ANALYSIS ===")
print("Rule violation distribution:")
print(df['rule_violation'].value_counts())
print(f"Rule violation rate: {df['rule_violation'].mean():.3f}")

# 3. ADVANCED DATA PREPROCESSING
print("\n=== ADVANCED DATA PREPROCESSING ===")

# Existing numeric features
numeric_features = ['body_len', 'Length_char', 'Length_word', 'NegWords', 
                   'UpperCaseRatio', 'ContainsLink', 'SentimentScore', 
                   'negScore', 'neuScore', 'posScore']

available_numeric = [col for col in numeric_features if col in df.columns]
X_numeric = df[available_numeric].copy().fillna(0)

print(f"Available numeric features: {len(available_numeric)}")

# Categorical features
X_categorical = pd.DataFrame()
if 'subreddit' in df.columns:
    le_subreddit = LabelEncoder()
    X_categorical['subreddit_encoded'] = le_subreddit.fit_transform(df['subreddit'].astype(str))
    print(f"Subreddit categories: {len(le_subreddit.classes_)}")

# ADVANCED TEXT PROCESSING
print("\n=== ADVANCED TEXT PROCESSING ===")
if 'body' in df.columns:
    print("Processing body text with advanced features...")
    
    # 1. Advanced text features
    text_features = advanced_text_features(df['body'])
    print(f"Created {text_features.shape[1]} advanced text features")
    
    # 2. TF-IDF features (top 50 most important terms)
    print("Creating TF-IDF features...")
    tfidf_features, tfidf_vectorizer = create_tfidf_features(df['body'], max_features=50)
    
    if not tfidf_features.empty:
        print(f"Created {tfidf_features.shape[1]} TF-IDF features")
        # Combine all features
        X = pd.concat([X_numeric, X_categorical, text_features, tfidf_features], axis=1)
    else:
        print("TF-IDF failed, using only advanced text features")
        X = pd.concat([X_numeric, X_categorical, text_features], axis=1)
else:
    X = pd.concat([X_numeric, X_categorical], axis=1)

# Fill any remaining NaN
X = X.fillna(0)

# Target variable
y = df['rule_violation'].astype(int)

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Total features: {len(X.columns)}")

# Show some example text features
if 'body' in df.columns:
    print("\n=== TEXT FEATURE EXAMPLES ===")
    print("Sample of text features created:")
    text_cols = [col for col in X.columns if any(pattern in col for pattern in ['word_', 'char_', 'spam_', 'positive_', 'negative_'])]
    if text_cols:
        print(X[text_cols[:10]].describe())

# 4. FEATURE SCALING
print("\n=== FEATURE SCALING ===")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

# 5. SPLIT DATA
print("\n=== DATA SPLITTING ===")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# 6. LOGISTIC REGRESSION MODEL
print("\n=== LOGISTIC REGRESSION MODEL ===")
logit_model = LogisticRegression(random_state=42, max_iter=2000, solver='liblinear')
logit_model.fit(X_train, y_train)

# Predictions
y_pred = logit_model.predict(X_test)
y_pred_proba = logit_model.predict_proba(X_test)[:, 1]

# 7. MODEL EVALUATION
print("\n=== MODEL PERFORMANCE ===")
print("Classification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

auc_score = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {auc_score:.4f}")

# 8. ADVANCED FEATURE INTERPRETATION
print("\n=== ADVANCED FEATURE INTERPRETATION ===")

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logit_model.coef_[0],
    'Odds_Ratio': np.exp(logit_model.coef_[0]),
    'Abs_Coefficient': np.abs(logit_model.coef_[0])
}).sort_values('Abs_Coefficient', ascending=False)

print("Top 15 Most Important Features:")
top_features = coefficients.head(15)
for idx, row in top_features.iterrows():
    direction = "↗️ increases" if row['Coefficient'] > 0 else "↘️ decreases"
    print(f"{row['Feature']}: {direction} rule violation probability")
    print(f"   Coefficient: {row['Coefficient']:.4f}, Odds Ratio: {row['Odds_Ratio']:.4f}")

# Analyze different feature types
print("\n=== FEATURE TYPE ANALYSIS ===")
text_features_coef = coefficients[coefficients['Feature'].str.contains('word_|char_|spam_|positive_|negative_|exclamation|question|caps')]
tfidf_features_coef = coefficients[coefficients['Feature'].str.contains('tfidf_')]
numeric_features_coef = coefficients[coefficients['Feature'].isin(available_numeric)]

print(f"Text features impact (avg abs coef): {text_features_coef['Abs_Coefficient'].mean():.4f}")
if not tfidf_features_coef.empty:
    print(f"TF-IDF features impact (avg abs coef): {tfidf_features_coef['Abs_Coefficient'].mean():.4f}")
print(f"Numeric features impact (avg abs coef): {numeric_features_coef['Abs_Coefficient'].mean():.4f}")

# 9. VISUALIZATION
print("\n=== CREATING VISUALIZATIONS ===")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Top features
top_15 = coefficients.head(15)
axes[0, 0].barh(range(len(top_15)), top_15['Coefficient'])
axes[0, 0].set_yticks(range(len(top_15)))
axes[0, 0].set_yticklabels([f[:20] + '...' if len(f) > 20 else f for f in top_15['Feature']])
axes[0, 0].set_xlabel('Coefficient')
axes[0, 0].set_title('Top 15 Feature Coefficients')
axes[0, 0].grid(True, alpha=0.3)

# 2. Feature types comparison
feature_types = ['Text Features', 'TF-IDF Features', 'Numeric Features']
if tfidf_features_coef.empty:
    impacts = [text_features_coef['Abs_Coefficient'].mean(), 0, numeric_features_coef['Abs_Coefficient'].mean()]
else:
    impacts = [text_features_coef['Abs_Coefficient'].mean(), 
              tfidf_features_coef['Abs_Coefficient'].mean(),
              numeric_features_coef['Abs_Coefficient'].mean()]

axes[0, 1].bar(feature_types, impacts, color=['lightblue', 'lightgreen', 'lightcoral'])
axes[0, 1].set_ylabel('Average |Coefficient|')
axes[0, 1].set_title('Feature Type Impact Comparison')
axes[0, 1].tick_params(axis='x', rotation=45)

# 3. ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
axes[1, 0].plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.4f})')
axes[1, 0].plot([0, 1], [0, 1], color='red', linestyle='--', alpha=0.7)
axes[1, 0].set_xlabel('False Positive Rate')
axes[1, 0].set_ylabel('True Positive Rate')
axes[1, 0].set_title('ROC Curve')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 4. Confusion Matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_title('Confusion Matrix')

plt.tight_layout()
plt.savefig('advanced_economic_analysis_results.png', dpi=300, bbox_inches='tight')
print("Advanced visualization saved as 'advanced_economic_analysis_results.png'")

# 10. TEXT INSIGHTS
print("\n=== TEXT-BASED INSIGHTS ===")
text_insights = coefficients[coefficients['Feature'].str.contains('spam_|positive_|negative_|caps|exclamation|question|url|email')]

if not text_insights.empty:
    print("Key Text-based Rule Violation Indicators:")
    for idx, row in text_insights.head(10).iterrows():
        direction = "⚠️ RISK FACTOR" if row['Coefficient'] > 0 else "✅ PROTECTIVE FACTOR"
        print(f"• {row['Feature']}: {direction} (coef: {row['Coefficient']:.4f})")

print(f"\n=== ADVANCED ANALYSIS COMPLETED ===")
print(f"Model Accuracy: {logit_model.score(X_test, y_test):.4f}")
print(f"Total Features Used: {len(X.columns)}")
print(f"Advanced Text Processing: ✅ Enabled")
if not tfidf_features.empty:
    print(f"TF-IDF Features: ✅ {tfidf_features.shape[1]} features")
else:
    print(f"TF-IDF Features: ❌ Disabled")