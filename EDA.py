from pyspark.sql.functions import (
    avg, stddev, expr, count, col, min as spark_min, max as spark_max
)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def perform_eda(df, sample_size=10000):    
    print("\n1. BASIC STATISTICS")
    print("-" * 40)
    
    df_clean = df.filter(col("is_fraud").isNotNull())
    
    print("\nFraud Distribution:")
    class_dist = df_clean.groupBy("is_fraud").count()
    class_dist.show()
    
    class_pd = class_dist.toPandas()
    if len(class_pd) >= 2:
        total = class_pd['count'].sum()
        fraud_row = class_pd[class_pd['is_fraud'] == 1]
        if len(fraud_row) > 0:
            fraud_pct = (fraud_row['count'].values[0] / total) * 100
            print(f"\nFraud percentage: {fraud_pct:.2f}%")
            print(f"Non-fraud percentage: {100 - fraud_pct:.2f}%")
    elif len(class_pd) == 1:
        if class_pd['is_fraud'].iloc[0] == 0:
            print("\nFraud percentage: 0.00%")
            print("Non-fraud percentage: 100.00%")
        else:
            print("\nFraud percentage: 100.00%")
            print("Non-fraud percentage: 0.00%")
    
    print("\nTransaction Amount Statistics:")
    df_clean.select("amt").summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max").show()
    
    if "distance_km" in df_clean.columns:
        print("\nDistance Statistics:")
        df_distance = df_clean.filter(
            (col("distance_km").isNotNull()) & 
            (col("distance_km") != float('inf')) &
            (col("distance_km") < 1e6)
        )
        df_distance.select("distance_km").summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max").show()
    
   
    print("\n2. PREPARING SAMPLE DATA")
    print("-" * 40)
    
    total_rows = df_clean.count()
    sample_fraction = min(1.0, sample_size / total_rows) if total_rows > 0 else 1.0
    
    print(f"Total rows: {total_rows:,}")
    print(f"Sampling {sample_size:,} rows for analysis")
    
    if "distance_km" in df_clean.columns:
        df_clean = df_clean.filter(
            (col("distance_km").isNotNull()) & 
            (col("distance_km") != float('inf')) &
            (col("distance_km") < 1e6)
        )
    
    pdf = df_clean.sample(fraction=sample_fraction, seed=42).limit(sample_size).toPandas()
    print(f"Sampled {len(pdf):,} rows")
    
    
    print("\n3. CORRELATION ANALYSIS")
    print("-" * 40)
    
    important_features = ['amt', 'hour', 'minute', 'is_night', 'is_business_hours', 'is_fraud']
    # if 'user_transaction_count' in pdf.columns:
    #     important_features.append('user_transaction_count')
    # if 'amt_zscore' in pdf.columns:
    #     important_features.append('amt_zscore')
    
    numeric_cols = [c for c in important_features if c in pdf.columns]
    
    if len(numeric_cols) > 1:
        pdf_clean = pdf[numeric_cols].dropna()
        
        if len(pdf_clean) > 0:
            corr_matrix = pdf_clean.corr()
            
            print("\nCorrelation Matrix:")
            print(corr_matrix.round(3))
            
            if "is_fraud" in corr_matrix.columns:
                print("\nTop correlations with fraud:")
                fraud_corr = corr_matrix["is_fraud"].drop("is_fraud").dropna()
                fraud_corr_sorted = fraud_corr.abs().sort_values(ascending=False)
                for feature in fraud_corr_sorted.head(5).index:
                    print(f"   {feature}: {fraud_corr[feature]:.3f}")
        else:
            print("Not enough clean data for correlation analysis")
    else:
        print("Not enough numeric columns for correlation analysis")
    
    
    print("\n4. SUMMARY STATISTICS BY FRAUD STATUS")
    print("-" * 40)
    
    if "is_fraud" in pdf.columns and len(pdf) > 0:
        agg_dict = {}
        if "amt" in pdf.columns:
            agg_dict["amt"] = ["mean", "median", "std", "count"]
        if "distance_km" in pdf.columns:
            pdf_dist = pdf[pdf["distance_km"] < 1e6]
            if len(pdf_dist) > 0:
                agg_dict["distance_km"] = ["mean", "median"]
        
        if agg_dict:
            fraud_stats = pdf.groupby("is_fraud").agg(agg_dict).round(2)
            print("\nStatistics by Fraud Status:")
            print(fraud_stats)
        else:
            print("No numeric columns available for summary statistics")
    
  
    print("\n5. ADDITIONAL INSIGHTS")
    print("-" * 40)
    
    if "is_fraud" in pdf.columns:
        if "hour" in pdf.columns:
            print("\nFraud Percentage by Hour (Top 5):")
            fraud_by_hour = pdf.groupby("hour")["is_fraud"].mean().sort_values(ascending=False)
            for hour, pct in fraud_by_hour.head(5).items():
                print(f"   Hour {int(hour)}: {pct*100:.2f}% fraud")
        
        if "is_night" in pdf.columns:
            print("\nFraud by Time of Day:")
            night_fraud = pdf[pdf["is_night"] == 1]["is_fraud"].mean() * 100 if len(pdf[pdf["is_night"] == 1]) > 0 else 0
            day_fraud = pdf[pdf["is_night"] == 0]["is_fraud"].mean() * 100 if len(pdf[pdf["is_night"] == 0]) > 0 else 0
            print(f"   Night (22:00-5:00): {night_fraud:.2f}% fraud")
            print(f"   Day (others): {day_fraud:.2f}% fraud")
        
        if "amt" in pdf.columns:
            amt_median = pdf["amt"].median()
            high_amt_fraud = pdf[pdf["amt"] > amt_median]["is_fraud"].mean() * 100 if len(pdf[pdf["amt"] > amt_median]) > 0 else 0
            low_amt_fraud = pdf[pdf["amt"] <= amt_median]["is_fraud"].mean() * 100 if len(pdf[pdf["amt"] <= amt_median]) > 0 else 0
            print(f"\nFraud by Amount:")
            print(f"   High amount (> ${amt_median:.2f}): {high_amt_fraud:.2f}% fraud")
            print(f"   Low amount (≤ ${amt_median:.2f}): {low_amt_fraud:.2f}% fraud")
        
        # print("\n" + "="*50)
        # print("KEY FINDINGS SUMMARY")
        # print("="*50)
        # print("✓ Fraud accounts for 12.77% of all transactions")
        # print("✓ Average fraud amount ($517.96) is 7.7x higher than normal ($66.81)")
        # print("✓ 42.3% of transactions at 11 PM are fraudulent")
        # print("✓ Night transactions (10 PM - 5 AM): 29.9% fraud rate vs 2.7% during day")
        # print("✓ High amount transactions: 19.8% fraud rate vs 5.8% for low amounts")
        # print("✓ Strongest predictor: transaction amount (correlation: 0.651)")
    
    return pdf
