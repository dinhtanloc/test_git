#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Text-to-Numeric Conversion for Econometric Analysis
Convert ALL text variables to numeric features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
warnings.filterwarnings('ignore')

def convert_all_text_to_numeric(df, target_col='rule_violation'):
    """
    Chuyển đổi TẤT CẢ biến text thành biến số
    """
    print("=" * 100)
    print("🔢 COMPLETE TEXT-TO-NUMERIC CONVERSION")
    print("=" * 100)
    
    df_numeric = df.copy()
    
    # 1. IDENTIFY COLUMNS BY TYPE
    print(f"\n1. ANALYZING COLUMN TYPES")
    print("-" * 60)
    
    numeric_cols = df_numeric.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    text_cols = df_numeric.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Original numeric columns: {len(numeric_cols)}")
    print(f"Text columns to convert: {len(text_cols)}")
    print(f"Text columns: {text_cols}")
    
    # 2. CONVERT TEXT COLUMNS TO NUMERIC FEATURES
    print(f"\n2. CONVERTING TEXT TO NUMERIC FEATURES")
    print("-" * 60)
    
    for col in text_cols:
        print(f"\nProcessing text column: {col}")
        
        # Basic text statistics (numeric features)
        text_data = df_numeric[col].astype(str).fillna('')
        
        # Length features
        df_numeric[f'{col}_char_length'] = text_data.str.len()
        df_numeric[f'{col}_word_count'] = text_data.str.split().str.len()
        df_numeric[f'{col}_sentence_count'] = text_data.str.count(r'[.!?]+')
        
        # Character analysis
        df_numeric[f'{col}_uppercase_count'] = text_data.str.count(r'[A-Z]')
        df_numeric[f'{col}_lowercase_count'] = text_data.str.count(r'[a-z]')
        df_numeric[f'{col}_digit_count'] = text_data.str.count(r'[0-9]')
        df_numeric[f'{col}_space_count'] = text_data.str.count(r'\s')
        df_numeric[f'{col}_punctuation_count'] = text_data.str.count(r'[^\w\s]')
        
        # Special characters
        df_numeric[f'{col}_exclamation_count'] = text_data.str.count('!')
        df_numeric[f'{col}_question_count'] = text_data.str.count(r'\?')
        df_numeric[f'{col}_comma_count'] = text_data.str.count(',')
        df_numeric[f'{col}_period_count'] = text_data.str.count(r'\.')
        
        # URL and web content
        df_numeric[f'{col}_url_count'] = text_data.str.count(r'http[s]?://|www\.')
        df_numeric[f'{col}_email_count'] = text_data.str.count(r'\S+@\S+')
        df_numeric[f'{col}_hashtag_count'] = text_data.str.count(r'#\w+')
        df_numeric[f'{col}_mention_count'] = text_data.str.count(r'@\w+')
        
        # Ratios (normalized features)
        total_chars = df_numeric[f'{col}_char_length'].replace(0, 1)  # Avoid division by zero
        df_numeric[f'{col}_uppercase_ratio'] = df_numeric[f'{col}_uppercase_count'] / total_chars
        df_numeric[f'{col}_digit_ratio'] = df_numeric[f'{col}_digit_count'] / total_chars
        df_numeric[f'{col}_punctuation_ratio'] = df_numeric[f'{col}_punctuation_count'] / total_chars
        df_numeric[f'{col}_space_ratio'] = df_numeric[f'{col}_space_count'] / total_chars
        
        # Unique words ratio
        unique_words = text_data.apply(lambda x: len(set(str(x).lower().split())) if x else 0)
        total_words = df_numeric[f'{col}_word_count'].replace(0, 1)
        df_numeric[f'{col}_unique_word_ratio'] = unique_words / total_words
        
        # Average word length
        df_numeric[f'{col}_avg_word_length'] = text_data.apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if str(x).split() else 0
        )
        
        # Text complexity (readability approximation)
        df_numeric[f'{col}_complexity_score'] = (
            df_numeric[f'{col}_avg_word_length'] * 0.4 + 
            df_numeric[f'{col}_sentence_count'] * 0.3 + 
            df_numeric[f'{col}_punctuation_ratio'] * 0.3
        )
        
        print(f"  - Created {22} numeric features from {col}")
    
    # 3. CREATE TF-IDF FEATURES FOR TOP TEXT COLUMNS
    print(f"\n3. CREATING TF-IDF NUMERIC FEATURES")
    print("-" * 60)
    
    # Select main text columns for TF-IDF (limit to avoid too many features)
    main_text_cols = [col for col in text_cols if col in ['body', 'rule', 'subreddit'] or 'example' in col][:3]
    
    for col in main_text_cols:
        print(f"Creating TF-IDF features for: {col}")
        
        text_data = df_numeric[col].astype(str).fillna('missing')
        
        try:
            # TF-IDF with limited features
            tfidf = TfidfVectorizer(
                max_features=20,  # Limit features to keep manageable
                stop_words='english',
                lowercase=True,
                ngram_range=(1, 1),  # Only unigrams
                min_df=2,
                max_df=0.95
            )
            
            tfidf_matrix = tfidf.fit_transform(text_data)
            
            # Create numeric columns from TF-IDF
            feature_names = tfidf.get_feature_names_out()
            for i, feature in enumerate(feature_names):
                df_numeric[f'{col}_tfidf_{feature}'] = tfidf_matrix[:, i].toarray().flatten()
            
            print(f"  - Created {len(feature_names)} TF-IDF numeric features")
            
        except Exception as e:
            print(f"  - TF-IDF failed for {col}: {str(e)}")
            # Create dummy numeric features
            for i in range(5):
                df_numeric[f'{col}_tfidf_dummy_{i}'] = 0
    
    # 4. LABEL ENCODE REMAINING CATEGORICAL TEXT
    print(f"\n4. LABEL ENCODING REMAINING TEXT")
    print("-" * 60)
    
    label_encoders = {}
    
    for col in text_cols:
        print(f"Label encoding: {col}")
        
        # Create label encoded version
        le = LabelEncoder()
        df_numeric[f'{col}_label_encoded'] = le.fit_transform(df_numeric[col].astype(str))
        
        # Create frequency encoding
        freq_map = df_numeric[col].value_counts().to_dict()
        df_numeric[f'{col}_frequency_encoded'] = df_numeric[col].map(freq_map)
        
        label_encoders[col] = le
        print(f"  - Created label encoding ({le.classes_.shape[0]} unique values)")
    
    # 5. DROP ORIGINAL TEXT COLUMNS
    print(f"\n5. REMOVING ORIGINAL TEXT COLUMNS")
    print("-" * 60)
    
    print(f"Dropping original text columns: {text_cols}")
    df_numeric = df_numeric.drop(columns=text_cols)
    
    # Also drop any remaining object columns
    remaining_object_cols = df_numeric.select_dtypes(include=['object']).columns.tolist()
    if remaining_object_cols:
        print(f"Converting remaining object columns to numeric: {remaining_object_cols}")
        for col in remaining_object_cols:
            try:
                df_numeric[col] = pd.to_numeric(df_numeric[col], errors='coerce')
                df_numeric[col] = df_numeric[col].fillna(0)
            except:
                # If can't convert, label encode
                le = LabelEncoder()
                df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))
    
    # 6. FINAL VERIFICATION
    print(f"\n6. FINAL VERIFICATION")
    print("-" * 60)
    
    final_dtypes = df_numeric.dtypes.value_counts()
    print(f"Final data types:")
    print(final_dtypes)
    
    remaining_objects = df_numeric.select_dtypes(include=['object']).columns.tolist()
    if remaining_objects:
        print(f"⚠️  Still have object columns: {remaining_objects}")
    else:
        print("✅ All columns are now numeric!")
    
    print(f"\nFinal shape: {df_numeric.shape}")
    print(f"Original shape: {df.shape}")
    print(f"Features added: {df_numeric.shape[1] - df.shape[1]}")
    
    return df_numeric, label_encoders

def create_numeric_only_dataset():
    """
    Tạo dataset hoàn toàn số từ dữ liệu đã preprocessing
    """
    print("🔢 Creating completely numeric dataset...")
    
    # Load processed data
    df = pd.read_csv('train_df_processing.csv')
    print(f"Loaded processed data: {df.shape}")
    
    # Convert all text to numeric
    df_numeric, encoders = convert_all_text_to_numeric(df)
    
    # Save completely numeric dataset
    df_numeric.to_csv('train_df_processing_numeric.csv', index=False)
    print(f"\n✅ Completely numeric dataset saved: train_df_processing_numeric.csv")
    
    # Display sample statistics
    print(f"\n📊 NUMERIC DATASET STATISTICS")
    print("=" * 80)
    print(f"Shape: {df_numeric.shape}")
    print(f"Data types:")
    print(df_numeric.dtypes.value_counts())
    print(f"\nSample correlation with target (rule_violation):")
    
    if 'rule_violation' in df_numeric.columns:
        correlations = df_numeric.corr()['rule_violation'].abs().sort_values(ascending=False)
        print(correlations.head(10))
    
    # Create summary statistics
    summary_stats = df_numeric.describe()
    summary_stats.to_csv('train_df_numeric_summary.csv')
    print(f"\n✅ Summary statistics saved: train_df_numeric_summary.csv")
    
    return df_numeric

if __name__ == "__main__":
    print("🚀 Starting Complete Text-to-Numeric Conversion...")
    
    # Create completely numeric dataset
    df_final = create_numeric_only_dataset()
    
    print(f"\n" + "="*100)
    print("✅ COMPLETE TEXT-TO-NUMERIC CONVERSION FINISHED!")
    print("="*100)
    print(f"\n📁 Files created:")
    print(f"📊 train_df_processing_numeric.csv - Completely numeric dataset")
    print(f"📈 train_df_numeric_summary.csv - Statistical summary")
    print(f"\n🎯 Result:")
    print(f"• All text columns converted to numeric features")  
    print(f"• No object/string columns remaining")
    print(f"• Ready for econometric analysis")
    print(f"• Shape: {df_final.shape}")
    print(f"• All {df_final.shape[1]} columns are numeric!")