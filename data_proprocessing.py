from pyspark.sql.functions import col, when, trim, regexp_replace
from pyspark.sql.types import StringType

def clean_string_columns(df):
    string_columns = ['merchant', 'job', 'category', 'city', 'state', 'trans_num', 'dob']
    
    for column in string_columns:
        if column in df.columns:
            df = df.withColumn(column, col(column).cast(StringType()))
            
            df = df.withColumn(column,
                trim(
                    regexp_replace(
                        regexp_replace(
                            regexp_replace(
                                regexp_replace(
                                    col(column),
                                    '\\\\', ''  
                                ),
                                '"', ''  
                            ),
                            '[\t\r\n]+', ' '  
                        ),
                        '\\s+', ' '  
                    )
                )
            )
            
            df = df.withColumn(column,
                when(col(column).isin('', 'null', 'NULL', 'None'), None)
                .otherwise(col(column))
            )    
    return df

def clean_merchant_column_specific(df):    
    if "merchant" in df.columns:        
        df = df.withColumn("merchant",
            regexp_replace(
                regexp_replace(
                    regexp_replace(
                        col("merchant"),
                        '[\\\\\\"\\t\\r\\n]+', ' '  
                    ),
                    '\\s+', ' '  
                ),
                '^\\s+|\\s+$', ''  
            )
        )
        
        df = df.withColumn("merchant",
            when(col("merchant").rlike('^[\\s]*$'), None)
            .when(col("merchant") == '', None)
            .when(col("merchant").isNull(), None)
            .otherwise(col("merchant"))
        )
    
    return df

def handle_missing_values(df):
    df = clean_string_columns(df)
    df = clean_merchant_column_specific(df)
    
    missing_counts = {}
    for column in df.columns:
        null_count = df.filter(col(column).isNull()).count()
        if null_count > 0:
            missing_counts[column] = null_count
            print(f"{column}: {null_count} missing values")
    
    if not missing_counts:
        print("No missing values found")
        return df
    
    essential_keywords = ["is_fraud", "amt"]
    essential_cols = [
        c for c in df.columns 
        if any(keyword in c.lower() for keyword in essential_keywords)
    ]
    
    if not essential_cols:
        print(f"No essential columns detected")
    else:
        print(f"Essential columns: {essential_cols}")
    
    df_clean = df.dropna(subset=essential_cols)
    dropped_rows = df.count() - df_clean.count()
    if dropped_rows > 0:
        print(f"Dropped {dropped_rows} rows with missing essential values")
    
    for column in df_clean.columns:
        if column in essential_cols:
            continue
            
        null_count = df_clean.filter(col(column).isNull()).count()
        if null_count == 0:
            continue
        
        col_type = str(df_clean.schema[column].dataType)
        
        if 'int' in col_type or 'double' in col_type or 'float' in col_type:
            median_value = df_clean.approxQuantile(column, [0.5], 0.01)[0]
            if median_value is not None:
                fill_value = median_value
                df_clean = df_clean.fillna({column: fill_value})
                print(f"{column}: filled {null_count} nulls with median = {fill_value:.2f}")
            else:
                df_clean = df_clean.fillna({column: 0})
                print(f"{column}: filled {null_count} nulls with 0")
        
        elif 'string' in col_type or 'char' in col_type:
            df_clean = df_clean.fillna({column: "unknown"})
            print(f"{column}: filled {null_count} nulls with 'unknown'")
        else:
            df_clean = df_clean.fillna({column: 0})
            print(f"{column}: filled {null_count} nulls with 0")


    remaining_nulls = 0
    for column in df_clean.columns:
        null_count = df_clean.filter(col(column).isNull()).count()
        if null_count > 0:
            remaining_nulls += null_count
            print(f"{column}: still has {null_count} nulls")
    
    if remaining_nulls == 0:
        print("All missing values handled successfully!")
    else:
        print(f"{remaining_nulls} total nulls remaining")    
    return df_clean

def balance_classes_undersample(df, fraud_col="is_fraud", seed=42):
  
    print("CLASS IMBALANCE HANDLING")
 
    fraud_df = df.filter(col(fraud_col) == 1)
    non_fraud_df = df.filter(col(fraud_col) == 0)
    
    fraud_count = fraud_df.count()
    non_fraud_count = non_fraud_df.count()
    
    print(f"   Fraud (1): {fraud_count} rows ({fraud_count/non_fraud_count*100:.2f}% of non-fraud)")
    print(f"   Non-Fraud (0): {non_fraud_count} rows")
    
    if fraud_count == 0:
        print("No fraud samples found")
        return df
    
    fraction = fraud_count / non_fraud_count
    non_fraud_sample = non_fraud_df.sample(fraction=fraction, seed=seed)
    
    balanced_df = fraud_df.union(non_fraud_sample)
    
    print(f"\nAfter undersampling:")
    print(f"   Fraud (1): {fraud_count} rows")
    print(f"   Non-Fraud (0): {non_fraud_sample.count()} rows")
    print(f"   Total: {balanced_df.count()} rows")
    
    return balanced_df

def balance_classes_oversample(df, fraud_col="is_fraud", seed=42):
    print("CLASS IMBALANCE HANDLING (OVERSAMPLING)")
    
    fraud_df = df.filter(col(fraud_col) == 1)
    non_fraud_df = df.filter(col(fraud_col) == 0)
    
    fraud_count = fraud_df.count()
    non_fraud_count = non_fraud_df.count()
    
    print(f"   Fraud (1): {fraud_count} rows")
    print(f"   Non-Fraud (0): {non_fraud_count} rows")
    
    if fraud_count == 0:
        print("No fraud samples found!")
        return df
    
    multiplier = (non_fraud_count // fraud_count) + 1
    
    oversampled_fraud = fraud_df
    for i in range(multiplier - 1):
        oversampled_fraud = oversampled_fraud.union(fraud_df)
    
    oversampled_fraud = oversampled_fraud.limit(non_fraud_count)
    
    balanced_df = non_fraud_df.union(oversampled_fraud)
    
    print(f"\nAfter oversampling:")
    print(f"   Fraud (1): {oversampled_fraud.count()} rows")
    print(f"   Non-Fraud (0): {non_fraud_count} rows")
    print(f"   Total: {balanced_df.count()} rows")
    
    return balanced_df
