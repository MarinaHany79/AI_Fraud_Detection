from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, regexp_replace, trim, lit, regexp_extract, expr, to_timestamp
from pyspark.sql.types import DoubleType, IntegerType, StringType
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def load_data(file_path):
    spark0 = SparkSession.builder.appName("Fraud Detection System").getOrCreate()
    df = spark0.read.csv(file_path,
                         header=True,
                         inferSchema=False,  
                         quote='"',           
                         escape='"',
                         multiLine=False)
    return df

def cast_columns(df):
  
    if "is_fraud" in df.columns:
        df = df.withColumn("is_fraud", 
                          regexp_extract(col("is_fraud"), r'^([01])', 1))
        df = df.withColumn("is_fraud", col("is_fraud").cast("int"))
    
    if "trans_date_trans_time" in df.columns:
        df = df.withColumn("trans_date_trans_time",
                          trim(regexp_replace(col("trans_date_trans_time"), '"', "")))
    
    if "amt" in df.columns:
        df = df.withColumn("amt", col("amt").cast("float"))
    
    if "city_pop" in df.columns:
        df = df.withColumn("city_pop", 
                          regexp_extract(col("city_pop"), r'(\d+)', 1))
        df = df.withColumn("city_pop", col("city_pop").cast("int"))
    
    for c in ["lat", "long", "merch_lat", "merch_long"]:
        if c in df.columns:
            df = df.withColumn(c, col(c).cast("double"))
    
    return df