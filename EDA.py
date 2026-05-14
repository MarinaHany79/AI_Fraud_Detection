from pyspark.sql.functions import col
import pandas as pd
import numpy as np


def perform_eda(df, sample_size=10000):

    print("\n1. BASIC STATISTICS")
    print("_" * 40)

    df_clean = df.filter(col("is_fraud").isNotNull())
 
    print("\nFraud Distribution")

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
    print("\nTransaction Amount Statistics:")

    df_clean.select("amt").summary("count", "mean", "stddev", "min", "25%", "50%", "75%", "max").show())


    print("\n2. PREPARING SAMPLE DATA")

    total_rows = df_clean.count()
    sample_fraction = ( min(1.0, sample_size / total_rows) if total_rows > 0 else 1.0)
    print(f"Total rows: {total_rows:,}")
    print(f"Sampling {sample_size:,} rows")

    pdf = ( df_clean.sample(fraction=sample_fraction, seed=42).limit(sample_size).toPandas() )
    print(f"Sampled rows: {len(pdf):,}")

    print("\n3. CORRELATION ANALYSIS")
    print("_" * 40)

    important_features = ["amt","hour","minute","is_night","is_business_hours","user_transaction_count",
        "user_avg_amt","hours_since_last_trans","is_rapid_transaction","trans_count_last_24h","is_fraud"]

    numeric_cols = [ c for c in important_features if c in pdf.columns]

    if len(numeric_cols) > 1:
        # pdf_clean = pdf[numeric_cols].dropna()
        if len(pdf_clean) > 0:
            corr_matrix = pdf_clean.corr()
            print("\nCorrelation Matrix:")
            print(corr_matrix.round(3))

            if "is_fraud" in corr_matrix.columns:
                print("\nTop correlations with fraud:")

                fraud_corr = (corr_matrix["is_fraud"].drop("is_fraud").dropna())
                fraud_corr_sorted = (fraud_corr.abs() .sort_values(ascending=False))
                for feature in fraud_corr_sorted.head(5).index:
                    print(f"{feature}: "f"{fraud_corr[feature]:.3f}" )

    print("\n4. SUMMARY STATISTICS BY FRAUD STATUS")
    print("-" * 40)

    if "is_fraud" in pdf.columns and len(pdf) > 0:
        agg_dict = {}
        if "amt" in pdf.columns:
            agg_dict["amt"] = ["mean","median","std","count"]

        if "user_avg_amt" in pdf.columns:
            agg_dict["user_avg_amt"] = [ "mean","median" ]

        if "user_transaction_count" in pdf.columns:
            agg_dict["user_transaction_count"] = [ "mean", "median"]

        if "trans_count_last_24h" in pdf.columns:
            agg_dict["trans_count_last_24h"] = ["mean","median" ]
        fraud_stats = ( pdf.groupby("is_fraud").agg(agg_dict).round(2) )
        print(fraud_stats)

    print("\n5.INSIGHTS")
    
    if "hour" in pdf.columns:
        print("\nFraud Percentage by Hour:")
        
        fraud_by_hour = ( pdf.groupby("hour")["is_fraud"] .mean() .sort_values(ascending=False) )

        for hour, pct in fraud_by_hour.head(5).items():
            print(f"Hour {int(hour)}: "f"{pct * 100:.2f}% fraud")

 
    if "is_night" in pdf.columns:
        print("\nFraud by Time of Day:")

        night_fraud = ( pdf[pdf["is_night"] == 1]["is_fraud"] .mean() * 100 )
        day_fraud = ( pdf[pdf["is_night"] == 0]["is_fraud"] .mean() * 100)
        print(f"Night: {night_fraud:.2f}% fraud")
        print(f"Day: {day_fraud:.2f}% fraud")

    
    if "amt" in pdf.columns:
        amt_median = pdf["amt"].median()
        high_amt_fraud = ( pdf[pdf["amt"] > amt_median]["is_fraud"] .mean() * 100 )
        low_amt_fraud = ( pdf[pdf["amt"] <= amt_median]["is_fraud"] .mean() * 100 )

        print("\nFraud by Amount:")
        print(f"High Amount: {high_amt_fraud:.2f}% fraud")
        print(f"Low Amount: {low_amt_fraud:.2f}% fraud")
   
    if "user_transaction_count" in pdf.columns:
        freq_median = pdf["user_transaction_count"].median()
        high_freq_fraud = ( pdf[pdf["user_transaction_count"] > freq_median]["is_fraud"].mean() * 100 )
        low_freq_fraud = (pdf[pdf["user_transaction_count"] <= freq_median]["is_fraud"].mean() * 100)

        print("\nFraud by Transaction Frequency:")
        print(f"High Frequency: {high_freq_fraud:.2f}% fraud")
        print(f"Low Frequency: {low_freq_fraud:.2f}% fraud")

    
    if "is_rapid_transaction" in pdf.columns:
        rapid_fraud = ( pdf[pdf["is_rapid_transaction"] == 1]["is_fraud"] .mean() * 100)
        normal_fraud = ( pdf[pdf["is_rapid_transaction"] == 0]["is_fraud"] .mean() * 100)

        print("\nFraud by Transaction Speed:")
        print(f"Rapid Transactions: {rapid_fraud:.2f}% fraud")
        print(f"Normal Transactions: {normal_fraud:.2f}% fraud")
