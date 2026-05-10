from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql.functions import col
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, test_data, model_name="Model"):
    print(f"\nEvaluating {model_name}")
    predictions = model.transform(test_data)
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="is_fraud", 
        predictionCol="prediction", 
        metricName="precisionByLabel"
    )
    precision = evaluator_precision.evaluate(predictions, {evaluator_precision.metricLabel: 1.0})
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="is_fraud", 
        predictionCol="prediction", 
        metricName="recallByLabel"
    )
    recall = evaluator_recall.evaluate(predictions, {evaluator_recall.metricLabel: 1.0})
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="is_fraud", 
        predictionCol="prediction", 
        metricName="f1"
    )
    f1 = evaluator_f1.evaluate(predictions)
    
    evaluator_accuracy = MulticlassClassificationEvaluator(
        labelCol="is_fraud", 
        predictionCol="prediction", 
        metricName="accuracy"
    )
    accuracy = evaluator_accuracy.evaluate(predictions)
    
    evaluator_auc = BinaryClassificationEvaluator(
        labelCol="is_fraud", 
        rawPredictionCol="rawPrediction", 
        metricName="areaUnderROC"
    )
    auc = evaluator_auc.evaluate(predictions)
    

    print(f"{model_name} RESULTS")
    print(f"Accuracy:  {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall:    {recall}")
    print(f"F1-Score:  {f1}")
    print(f"ROC-AUC:   {auc}")
    
    cm = predictions.groupBy("is_fraud", "prediction").count().toPandas()
    tn = cm[(cm['is_fraud']==0) & (cm['prediction']==0)]['count'].values[0] if len(cm[(cm['is_fraud']==0) & (cm['prediction']==0)]) > 0 else 0
    fp = cm[(cm['is_fraud']==0) & (cm['prediction']==1)]['count'].values[0] if len(cm[(cm['is_fraud']==0) & (cm['prediction']==1)]) > 0 else 0
    fn = cm[(cm['is_fraud']==1) & (cm['prediction']==0)]['count'].values[0] if len(cm[(cm['is_fraud']==1) & (cm['prediction']==0)]) > 0 else 0
    tp = cm[(cm['is_fraud']==1) & (cm['prediction']==1)]['count'].values[0] if len(cm[(cm['is_fraud']==1) & (cm['prediction']==1)]) > 0 else 0
    
    print(f"\nConfusion Matrix:")
    print(f"                 Actual")
    print(f"              Non-Fraud  Fraud")
    print(f"Predicted Non-Fraud   {tn:>6}   {fn:>6}")
    print(f"         Fraud         {fp:>6}   {tp:>6}")
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': auc,
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp
    }
    
    return metrics, predictions


def compare_models(models_dict, test_data):
    all_metrics = []
    all_predictions = {}
    
    for model_name, model in models_dict.items():
        metrics, predictions = evaluate_model(model, test_data, model_name)
        all_metrics.append(metrics)
        all_predictions[model_name] = predictions

    print("MODEL COMPARISON TABLE")
    comparison_df = pd.DataFrame(all_metrics)
    comparison_df = comparison_df[['model', 'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
    comparison_df = comparison_df.round(4)
    print(comparison_df.to_string(index=False))
    
    best_accuracy = max(all_metrics, key=lambda x: x['accuracy'])
    best_f1 = max(all_metrics, key=lambda x: x['f1_score'])
    best_auc = max(all_metrics, key=lambda x: x['roc_auc'])
    
    print(f"\nBest by Accuracy: {best_accuracy['model']} (Accuracy = {best_accuracy['accuracy']:.4f})")
    print(f"Best by F1-Score: {best_f1['model']} (F1 = {best_f1['f1_score']:.4f})")
    print(f"Best by ROC-AUC: {best_auc['model']} (AUC = {best_auc['roc_auc']:.4f})")
    return all_metrics, all_predictions

def plot_comparison(metrics_list):
    df = pd.DataFrame(metrics_list)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(df['model']))
    width = 0.25
    
    ax.bar(x - width, df['accuracy'], width, label='Accuracy', color='#3498db')
    ax.bar(x, df['f1_score'], width, label='F1-Score', color='#2ecc71')
    ax.bar(x + width, df['roc_auc'], width, label='ROC-AUC', color='#e74c3c')
    
    ax.set_title('Model Comparison: Accuracy vs F1-Score vs ROC-AUC', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df['model'], fontsize=11)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Score', fontsize=12)
    ax.legend(loc='lower right', fontsize=11)
    
    for i, row in df.iterrows():
        ax.text(i - width, row['accuracy'] + 0.02, f'{row["accuracy"]:.3f}', ha='center', fontsize=9)
        ax.text(i, row['f1_score'] + 0.02, f'{row["f1_score"]:.3f}', ha='center', fontsize=9)
        ax.text(i + width, row['roc_auc'] + 0.02, f'{row["roc_auc"]:.3f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def quick_evaluation(model, test_data):
    
    predictions = model.transform(test_data)
    
    evaluator_acc = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction", metricName="accuracy")
    accuracy = evaluator_acc.evaluate(predictions)
    
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="is_fraud", predictionCol="prediction", metricName="f1")
    f1 = evaluator_f1.evaluate(predictions)
    
    evaluator_auc = BinaryClassificationEvaluator(labelCol="is_fraud", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    auc = evaluator_auc.evaluate(predictions)
    
    print(f"Accuracy: {accuracy:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {auc:.4f}")
    
    return accuracy, f1, auc