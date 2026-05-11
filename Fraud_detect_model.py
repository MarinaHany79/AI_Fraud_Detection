from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier
from pyspark.sql.functions import col

def split_data(df, train_ratio=0.7, test_ratio=0.3, seed=42):
    train_df, test_df = df.randomSplit([train_ratio, test_ratio], seed=seed)
    train_fraud = train_df.filter(col("is_fraud") == 1).count()
    test_fraud = test_df.filter(col("is_fraud") == 1).count()
    train_total = train_df.count()
    test_total = test_df.count()
    
    print(f"Training set: {train_total:,} rows ({train_ratio*100:.0f}%)")
    print(f"Test set: {test_total:,} rows ({test_ratio*100:.0f}%)")
    print(f"\n Fraud distribution:")
    print(f" Train - Fraud: {train_fraud} ({train_fraud/train_total*100:.2f}%)")
    print(f" Test  - Fraud: {test_fraud} ({test_fraud/test_total*100:.2f}%)")
    return train_df, test_df

def train_logistic_regression(train_data, feature_col="scaled_features"):
    lr = LogisticRegression( featuresCol=feature_col, labelCol="is_fraud",maxIter=50,regParam=0.01, elasticNetParam=0.5)
    model = lr.fit(train_data)
    print("Logistic Regression trained successfully!")
    return model

def train_random_forest(train_data, feature_col="scaled_features"):
 
    rf = RandomForestClassifier(featuresCol=feature_col,labelCol="is_fraud",numTrees=50,maxDepth=10,seed=42)
    model = rf.fit(train_data)
    print(f" Random Forest trained successfully! ({model.numTrees} trees)")
    return model

def train_gradient_boosting(train_data, feature_col="scaled_features"):
   
    gbt = GBTClassifier(featuresCol=feature_col,labelCol="is_fraud",maxIter=50,maxDepth=5,stepSize=0.1,seed=42)
    model = gbt.fit(train_data)
    print("Gradient Boosting trained successfully!")
    return model

def train_all_models(train_data):
    print("STARTING MODEL TRAINING")    
    models = {}
    models['Logistic Regression'] = train_logistic_regression(train_data)
    models['Random Forest'] = train_random_forest(train_data)
    models['Gradient Boosting'] = train_gradient_boosting(train_data)
    
    return models
