from pyspark.sql.functions import (
    avg, stddev, lag, acos, sin, cos, radians, 
    lit, count, when, col, 
    trim, regexp_replace, abs, split, 
    concat_ws, coalesce, regexp_extract
)
from pyspark.sql import Window

def engineer_features(df):
    
    print("Extracting date and time components")
    
    if "trans_date_trans_time" in df.columns:
        df = df.withColumn("date_clean", 
                          trim(regexp_replace(col("trans_date_trans_time"), '"', '')))
        
        df = df.withColumn("date_part", split(col("date_clean"), " ")[0])
        df = df.withColumn("time_part", split(col("date_clean"), " ")[1])
        
        df = df.withColumn("year", 
            regexp_extract(col("date_part"), r'(19|20)\d{2}', 0).cast("int"))
        
        df = df.withColumn("month_slash", 
            when(col("date_part").contains("/"), split(col("date_part"), "/")[0]))
        df = df.withColumn("day_slash", 
            when(col("date_part").contains("/"), split(col("date_part"), "/")[1]))
        
        df = df.withColumn("day_dash", 
            when(col("date_part").contains("-"), split(col("date_part"), "-")[0]))
        df = df.withColumn("month_dash", 
            when(col("date_part").contains("-"), split(col("date_part"), "-")[1]))
        
        df = df.withColumn("month", 
            coalesce(col("month_slash"), col("month_dash")).cast("int"))
        df = df.withColumn("day", 
            coalesce(col("day_slash"), col("day_dash")).cast("int"))
        
        df = df.withColumn("hour_str", split(col("time_part"), ":")[0])
        df = df.withColumn("minute_str", split(col("time_part"), ":")[1])
        
        df = df.withColumn("hour", col("hour_str").cast("int"))
        df = df.withColumn("minute", col("minute_str").cast("int"))
        
        df = df.fillna({"hour": 0, "minute": 0, "month": 1, "day": 1})
    
    before_count = df.count()
    df = df.filter(col("year").isNotNull() & col("month").isNotNull() & col("day").isNotNull())
    after_count = df.count()
    
    if before_count > after_count:
        print(f"  Dropped {before_count - after_count:,} rows with invalid dates")
    print(f" Kept {after_count:,} rows")
    
    print("  Creating time-based features...")
    
    df = df.withColumn("is_night", 
        when((col("hour") >= 22) | (col("hour") <= 5), 1).otherwise(0))
    df = df.withColumn("is_business_hours", 
        when((col("hour") >= 9) & (col("hour") <= 17), 1).otherwise(0))
    df = df.withColumn("day_of_month", col("day"))
    
    df = df.withColumn("year_long", col("year").cast("long"))
    df = df.withColumn("month_long", col("month").cast("long"))
    df = df.withColumn("day_long", col("day").cast("long"))
    df = df.withColumn("hour_long", col("hour").cast("long"))
    df = df.withColumn("minute_long", col("minute").cast("long"))
    
    df = df.withColumn("trans_seconds",
        (col("year_long") * 100000000 + 
         col("month_long") * 1000000 + 
         col("day_long") * 10000 + 
         col("hour_long") * 100 + 
         col("minute_long")).cast("long"))
    
    print("  Creating distance features...")
    
    if all(c in df.columns for c in ["lat", "long", "merch_lat", "merch_long"]):
        df = df.withColumn("lat", col("lat").cast("double"))
        df = df.withColumn("long", col("long").cast("double"))
        df = df.withColumn("merch_lat", col("merch_lat").cast("double"))
        df = df.withColumn("merch_long", col("merch_long").cast("double"))
        
        df = df.fillna({"lat": 0.0, "long": 0.0, "merch_lat": 0.0, "merch_long": 0.0})
        
        df = df.withColumn("distance_km",
            acos(
                sin(radians(col("lat"))) * sin(radians(col("merch_lat"))) +
                cos(radians(col("lat"))) * cos(radians(col("merch_lat"))) *
                cos(radians(col("merch_long")) - radians(col("long")))
            ) * 6371
        )
        df = df.fillna({"distance_km": 0})
        df = df.withColumn("is_long_distance", when(col("distance_km") > 100, 1).otherwise(0))
        df = df.withColumn("is_very_long_distance", when(col("distance_km") > 500, 1).otherwise(0))
    
    if "amt" in df.columns:
        df = df.withColumn("amt", col("amt").cast("double"))
        df = df.fillna({"amt": 0.0})
   
    if "trans_num" in df.columns:
        df = df.withColumn("trans_num", 
                          trim(regexp_replace(col("trans_num"), '"', '')))
    
    if "cc_num" in df.columns:
        df = df.withColumn("cc_num", 
                          trim(regexp_replace(col("cc_num"), '"', '')))
   
    print("  Creating user-based features...")
    
    user_col = "cc_num" if "cc_num" in df.columns else "trans_num"
    
    if user_col in df.columns:
        user_window = Window.partitionBy(user_col)
        
        df = df.withColumn("user_transaction_count", count("*").over(user_window))
        df = df.withColumn("user_avg_amt", avg("amt").over(user_window))
        df = df.withColumn("user_std_amt", stddev("amt").over(user_window))
    
    print("  Creating velocity features...")
    
    if user_col in df.columns:
        user_time_window = Window.partitionBy(user_col).orderBy("trans_seconds")
        df = df.withColumn("prev_seconds", lag("trans_seconds").over(user_time_window))
        df = df.withColumn("hours_since_last_trans",
            when(col("prev_seconds").isNotNull(),
                 (col("trans_seconds") - col("prev_seconds")) / 10000.0
            ).otherwise(-1))
        df = df.withColumn("is_rapid_transaction",
            when(col("hours_since_last_trans").between(0, 1), 1).otherwise(0))
        
        last_24h_window = Window.partitionBy(user_col).orderBy("trans_seconds").rangeBetween(-10000, 0)
        df = df.withColumn("trans_count_last_24h", count("*").over(last_24h_window))
    
    if "category" in df.columns and user_col in df.columns:
        print("  Creating category-based features...")
        user_cat_window = Window.partitionBy(user_col, "category")
        df = df.withColumn("user_category_count", count("*").over(user_cat_window))
        df = df.withColumn("user_category_avg_amt", avg("amt").over(user_cat_window))
        df = df.withColumn("user_category_avg_amt", 
            when(col("user_category_avg_amt").isNull(), 0.0)
            .otherwise(col("user_category_avg_amt")))
    
    if "dob" in df.columns:
        print("  Creating age features...")
        
        df = df.withColumn("dob_clean", trim(regexp_replace(col("dob"), '"', "")))
        df = df.withColumn("birth_year", 
            regexp_extract(col("dob_clean"), r'(19|20)\d{2}', 0).cast("int"))
        
        df = df.withColumn("age", 
            when(col("birth_year").isNotNull() & col("year").isNotNull(),
                 col("year") - col("birth_year"))
            .otherwise(lit(None)))
        
        df = df.withColumn("age", 
            when(col("age").between(0, 100), col("age"))
            .otherwise(lit(None)))
        
        df = df.fillna({"age": 40})
        
        df = df.withColumn("age_group",
            when(col("age") < 25, "18-24")
            .when(col("age") < 35, "25-34")
            .when(col("age") < 50, "35-49")
            .when(col("age") < 65, "50-64")
            .otherwise("65+"))
        
        df = df.drop("dob_clean", "birth_year")
   
    if "city" in df.columns and user_col in df.columns:
        print("  Creating city-based features...")
        user_city_window = Window.partitionBy(user_col, "city")
        df = df.withColumn("user_city_count", count("*").over(user_city_window))
        df = df.withColumn("is_new_city", when(col("user_city_count") == 1, 1).otherwise(0))
    
    df = df.drop("date_clean", "date_part", "time_part", "prev_seconds",
                 "month_slash", "day_slash", "day_dash", "month_dash",
                 "hour_str", "minute_str", "year_long", "month_long", 
                 "day_long", "hour_long", "minute_long")
    
    print("\n" + "=" * 50)
    print(f"  Total columns: {len(df.columns)}")
    print(f"  Final rows: {df.count():,}")
    print("=" * 50)
    
    return df
