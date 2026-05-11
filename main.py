from pyspark.sql.functions import col, when
from pyspark.ml.feature import VectorAssembler, StandardScaler
import pandas as pd

from load_dataset import load_data, cast_columns
from data_proprocessing import handle_missing_values, balance_classes_undersample, balance_classes_oversample ,clean_merchant_column_specific , clean_string_columns
from feature_engineering import engineer_features
from EDA import perform_eda
from Fraud_detect_model import split_data, train_all_models
from model_Evaluation import compare_models, draw_comparison,evalute_model

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
    
    print("\n 1.LOADING DATA")
    df_raw = load_data(file_path)
    print(f"   Loaded {df_raw.count():,} rows with {len(df_raw.columns)} columns")

    print("\n 2.HANDLING MISSING VALUES")
    df_clean = handle_missing_values(df_raw)
    print(f"   After cleaning: {df_clean.count():,} rows")
    
    print("\n 3.CASTING COLUMNS")
    df_typed = cast_columns(df_clean)
    print("   Data types converted successfully")
    
    print("\n 4.FEATURE ENGINEERING")
    df_features = engineer_features(df_typed)
    print(f"   Feature engineering complete: {len(df_features.columns)} total columns")
    
    if do_eda:
        print("\n 5.EXPLORATORY DATA ANALYSIS")
        perform_eda(df_features, sample_size=50000)
        print("   EDA complete")
        
    print("\n 6.PREPARING DATA FOR MODELING")
    selected_features = [ "amt", "hour", "minute", "is_night", "is_business_hours", "day_of_month", 
                         "user_transaction_count", "user_avg_amt","hours_since_last_trans","is_rapid_transaction",
                         "trans_count_last_24h"]
    selected_features = [
        c for c in selected_features
        if c in df_features.columns
    ]

    print(f"Selected Features ({len(selected_features)}):")
    for feature in selected_features:
        print(f"   - {feature}")
    for feature in selected_features:

        df_features = df_features.fillna({
            feature: 0.0
        })
    assembler = VectorAssembler(
        inputCols=selected_features,
        outputCol="features",
        handleInvalid="skip"
    )

    df_assembled = assembler.transform(df_features)
    df_assembled = df_assembled.filter(
        col("features").isNotNull()
    )
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=True
    )
    scaler_model = scaler.fit(df_assembled)
    df_scaled = scaler_model.transform(df_assembled)
    print("Feature preparation completed successfully")
    print(f"Using {len(selected_features)} features for modeling")
    print("\n 7.BALANCING CLASSES")
    
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
        print("\nSTEP 8: MODEL TRAINING & EVALUATION")
        
        train_df, test_df = split_data(df_balanced, train_ratio=0.7, test_ratio=0.3)
        print(f"   Train set: {train_df.count():,} rows")
        print(f"   Test set: {test_df.count():,} rows")
        
        models = train_all_models(train_df)
        print(f"   Trained {len(models)} models")
        
        all_metrics, all_predictions = compare_models( models, test_df )
        
       comparison_figure = draw_comparison(all_metrics)
        
       best_model_metrics = max( all_metrics, key=lambda x: x['f1_score'] ) 
       best_model_name = best_model_metrics['model'] 

       print("\n BEST MODEL") 
       print( f"Best Model Based on F1-Score: " f"{best_model_name}" ) 
       print( f"Accuracy : " f"{best_model_metrics['accuracy']:.4f}" ) 
       print( f"Precision: " f"{best_model_metrics['precision']:.4f}" ) 
       print( f"Recall : " f"{best_model_metrics['recall']:.4f}" )
       print( f"F1-Score : " f"{best_model_metrics['f1_score']:.4f}" ) 
       print( f"ROC-AUC : " f"{best_model_metrics['roc_auc']:.4f}" )

    if save_data_path:
        print("\n 9.SAVING PROCESSED DATA")
        cols_to_drop = ["features", "scaled_features"]
        cols_to_drop = [c for c in cols_to_drop if c in df_balanced.columns]
        df_to_save = df_balanced.drop(*cols_to_drop)
        save_data(df_to_save, save_data_path, format="csv")
    
    if save_model_path and best_model and models:
        print("\nSTEP 10: SAVING BEST MODEL")
        if best_model['model'] in models:
            best_model_obj = models[best_model['model']]
            model_filename = best_model['model'].replace(' ', '_')
            save_model(best_model_obj, f"{save_model_path}/{model_filename}")
    
    if save_metrics_path and metrics:
        print("\n 11.SAVING METRICS")
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
