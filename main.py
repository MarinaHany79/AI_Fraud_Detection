from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
import pandas as pd

from load_dataset import load_data, cast_columns
from data_proprocessing import handle_missing_values, balance_classes_undersample, balance_classes_oversample ,clean_merchant_column_specific , clean_string_columns
from feature_engineering import engineer_features
from EDA import perform_eda
from Fraud_detect_model import split_data, train_all_models
from model_Evaluation import compare_models, plot_comparison, quick_evaluation

def save_data(df, output_path, format="csv", mode="overwrite"):
    print(f"\nSaving data to: {output_path}")
    writer = df.write.format(format).mode(mode)
    if format == "csv":
        writer.option("header", "true")
    writer.save(output_path)
    print(f"Data saved successfully!")

def save_model(model, output_path):
    print(f"\nSaving model to: {output_path}")
    model.write().overwrite().save(output_path)
    print(f"Model saved successfully!")

def save_metrics(metrics, output_path):
    print(f"\nSaving metrics to: {output_path}")
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(output_path, index=False)
    print(f"Metrics saved successfully!")

def run_fraud_detection_pipeline(file_path, balance_method="undersample", do_eda=True, do_training=True, 
                                 save_data_path=None, save_model_path=None, save_metrics_path=None):
    
    print("\nSTEP 1: LOADING DATA")
    df_raw = load_data(file_path)
    print(f"   Loaded {df_raw.count():,} rows with {len(df_raw.columns)} columns")

    print("\nSTEP 2: Clean String Columns")
    df_clean_string = clean_string_columns(df_raw)
    df_clean_merchant= clean_merchant_column_specific(df_clean_string)
   
    print("\nSTEP 3: HANDLING MISSING VALUES")
    df_clean = handle_missing_values(df_raw)
    print(f"   After cleaning: {df_clean.count():,} rows")
    
    print("\nSTEP 4: CASTING COLUMNS")
    df_typed = cast_columns(df_clean)
    print("   Data types converted successfully")
    
    print("\nSTEP 5: FEATURE ENGINEERING")
    df_features = engineer_features(df_typed)
    print(f"   Feature engineering complete: {len(df_features.columns)} total columns")
    
    if do_eda:
        print("\nSTEP 6: EXPLORATORY DATA ANALYSIS")
        perform_eda(df_features, sample_size=50000)
        print("   EDA complete")
    
    print("\nSTEP 7: PREPARING DATA FOR MODELING")
    
    exclude_cols = ["is_fraud", "trans_datetime", "trans_date_trans_time", "dob", 
                    "dob_parsed", "prev_trans_time", "age_group", "trans_year", "birth_year"]
    exclude_cols = [c for c in exclude_cols if c in df_features.columns]
    
    numeric_cols = []
    for col_name, col_type in df_features.dtypes:
        if col_type in ('int', 'double', 'float', 'bigint') and col_name not in exclude_cols:
            numeric_cols.append(col_name)
    
    print(f"   Using {len(numeric_cols)} numeric features for modeling")
    
    for col_name in numeric_cols:
        df_features = df_features.fillna({col_name: 0.0})
    
    assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features", handleInvalid="skip")
    df_assembled = assembler.transform(df_features)
    df_assembled = df_assembled.filter(col("features").isNotNull())
    
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", 
                           withStd=True, withMean=True)
    scaler_model = scaler.fit(df_assembled)
    df_scaled = scaler_model.transform(df_assembled)
    
    print("\nSTEP 8: BALANCING CLASSES")
    
    if balance_method == "undersample":
        df_balanced = balance_classes_undersample(df_scaled, fraud_col="is_fraud")
        print(f"   Using undersampling")
    elif balance_method == "oversample":
        df_balanced = balance_classes_oversample(df_scaled, fraud_col="is_fraud")
        print(f"   Using oversampling")
    else:
        print(f"   Unknown balance method '{balance_method}', using original data")
        df_balanced = df_scaled
    
    print(f"   Balanced data: {df_balanced.count():,} rows")
    
    models = None
    metrics = None
    best_model = None
    
    if do_training:
        print("\nSTEP 9: MODEL TRAINING & EVALUATION")
        
        train_df, test_df = split_data(df_balanced, train_ratio=0.7, test_ratio=0.3)
        print(f"   Train set: {train_df.count():,} rows")
        print(f"   Test set: {test_df.count():,} rows")
        
        models = train_all_models(train_df)
        print(f"   Trained {len(models)} models")
        
        metrics, predictions = compare_models(models, test_df)
        
        try:
            plot_comparison(metrics)
            print("   Comparison plot generated")
        except Exception as e:
            print(f"   Could not generate plot: {e}")
        
        best_model = max(metrics, key=lambda x: x.get('f1_score', 0))
        
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"Best Model: {best_model['model']}")
        print(f"   - Accuracy:  {best_model.get('accuracy', 0):.4f}")
        print(f"   - F1-Score:  {best_model.get('f1_score', 0):.4f}")
        print(f"   - ROC-AUC:   {best_model.get('roc_auc', 0):.4f}")
        print(f"   - Precision: {best_model.get('precision', 0):.4f}")
        print(f"   - Recall:    {best_model.get('recall', 0):.4f}")
    
    if save_data_path:
        print("\nSTEP 10: SAVING PROCESSED DATA")
        cols_to_drop = ["features", "scaled_features"]
        cols_to_drop = [c for c in cols_to_drop if c in df_balanced.columns]
        df_to_save = df_balanced.drop(*cols_to_drop)
        save_data(df_to_save, save_data_path, format="csv")
    
    if save_model_path and best_model and models:
        print("\nSTEP 11: SAVING BEST MODEL")
        if best_model['model'] in models:
            best_model_obj = models[best_model['model']]
            model_filename = best_model['model'].replace(' ', '_')
            save_model(best_model_obj, f"{save_model_path}/{model_filename}")
    
    if save_metrics_path and metrics:
        print("\nSTEP 12: SAVING METRICS")
        save_metrics(metrics, save_metrics_path)

    return df_balanced, models, metrics

if __name__ == "__main__":
    file_path = "/Volumes/workspace/default/ai_fraud_detection/fraud_data.csv"
    save_data_path = "/Volumes/workspace/default/ai_fraud_detection/clean_fraud_data"
    save_model_path = "/Volumes/workspace/default/ai_fraud_detection/models"
    save_metrics_path = "/Volumes/workspace/default/ai_fraud_detection/metrics.csv"
    
    print("="*60)
    print("FRAUD DETECTION SYSTEM")
    print("="*60)
    
    try:
        df_final, models, metrics = run_fraud_detection_pipeline(
            file_path=file_path,
            balance_method="undersample",  
            do_eda=True,                  
            do_training=True,              
            save_data_path=save_data_path,  
            save_model_path=save_model_path, 
            save_metrics_path=save_metrics_path 
        )
        
        print("\n executed successfully ")
        
    except Exception as e:
        print(f"\nError : {e}")
        import traceback
        traceback.print_exc()