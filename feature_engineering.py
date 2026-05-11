from pyspark.sql.functions import (avg,count,lag,when,col,trim,regexp_replace,split,regexp_extract,coalesce)
from pyspark.sql import Window

def engineer_features(df):
    if "trans_date_trans_time" in df.columns:
        df = df.withColumn(
            "date_clean",
            trim(
                regexp_replace(
                    col("trans_date_trans_time"),
                    '"',
                    ''
                )
            )
        )
        
        df = df.withColumn("date_part",split(col("date_clean"), " ")[0])
        df = df.withColumn( "time_part",split(col("date_clean"), " ")[1] )
        df = df.withColumn("year", regexp_extract(col("date_part"),r'(19|20)\d{2}',0).cast("int"))
        df = df.withColumn("month_slash",when( col("date_part").contains("/"), split(col("date_part"), "/")[0]) )
        df = df.withColumn("day_slash",when(col("date_part").contains("/"),split(col("date_part"), "/")[1] ))
        df = df.withColumn("day_dash",when(col("date_part").contains("-"),split(col("date_part"), "-")[0]))
        df = df.withColumn( "month_dash", when(col("date_part").contains("-"),split(col("date_part"), "-")[1]))\
        
        df = df.withColumn("month",coalesce(col("month_slash"),col("month_dash")).cast("int"))
        df = df.withColumn("day",coalesce(col("day_slash"),col("day_dash")).cast("int") )
        
        df = df.withColumn("hour", split(col("time_part"), ":")[0].cast("int"))
        df = df.withColumn("minute",split(col("time_part"), ":")[1].cast("int"))
        df = df.fillna({"hour": 0,"minute": 0,"month": 1,"day": 1})
        
    before_count = df.count()
    df = df.filter(col("year").isNotNull() &col("month").isNotNull() &col("day").isNotNull())
    after_count = df.count()
    if before_count > after_count:
        print(f"Dropped {before_count - after_count:,} rows "f"with invalid dates")

    print(f"Remaining rows: {after_count:,}")
    print("Creating time-based features.")
    df = df.withColumn("is_night",when((col("hour") >= 22) |(col("hour") <= 5),1).otherwise(0))
    df = df.withColumn("is_business_hours",
        when((col("hour") >= 9) &(col("hour") <= 17), 1 ).otherwise(0) )

    df = df.withColumn( "day_of_month", col("day"))

    df = df.withColumn(
        "year_long",
        col("year").cast("long")
    )

    df = df.withColumn(
        "month_long",
        col("month").cast("long")
    )

    df = df.withColumn(
        "day_long",
        col("day").cast("long")
    )

    df = df.withColumn(
        "hour_long",
        col("hour").cast("long")
    )

    df = df.withColumn(
        "minute_long",
        col("minute").cast("long")
    )

    df = df.withColumn(
        "trans_seconds",
        (
            col("year_long") * 100000000 +
            col("month_long") * 1000000 +
            col("day_long") * 10000 +
            col("hour_long") * 100 +
            col("minute_long")
        ).cast("long")
    )
    if "amt" in df.columns:

        df = df.withColumn(
            "amt",
            col("amt").cast("double")
        )

        df = df.fillna({
            "amt": 0.0
        })

    if "cc_num" in df.columns:

        df = df.withColumn(
            "cc_num",
            trim(
                regexp_replace(
                    col("cc_num"),
                    '"',
                    ''
                )
            )
        )

    if "trans_num" in df.columns:

        df = df.withColumn(
            "trans_num",
            trim(
                regexp_replace(
                    col("trans_num"),
                    '"',
                    ''
                )
            )
        )

    print("Creating user behavior features...")

    user_col = (
        "cc_num"
        if "cc_num" in df.columns
        else "trans_num"
    )

    if user_col in df.columns:

        user_window = Window.partitionBy(user_col)

        df = df.withColumn(
            "user_transaction_count",
            count("*").over(user_window)
        )

        df = df.withColumn(
            "user_avg_amt",
            avg("amt").over(user_window)
        )

    print("Creating transaction velocity features...")

    if user_col in df.columns:

        user_time_window = (
            Window
            .partitionBy(user_col)
            .orderBy("trans_seconds")
        )

        df = df.withColumn(
            "prev_seconds",
            lag("trans_seconds").over(user_time_window)
        )

        df = df.withColumn(
            "hours_since_last_trans",
            when(
                col("prev_seconds").isNotNull(),
                (
                    col("trans_seconds") -
                    col("prev_seconds")
                ) / 10000.0
            ).otherwise(-1)
        )

        df = df.withColumn(
            "is_rapid_transaction",
            when(
                col("hours_since_last_trans").between(0, 1),
                1
            ).otherwise(0)
        )

        last_window = (
            Window
            .partitionBy(user_col)
            .orderBy("trans_seconds")
            .rangeBetween(-10000, 0)
        )

        df = df.withColumn(
            "trans_count_last_24h",
            count("*").over(last_window)
        )

    df = df.drop(
        "date_clean",
        "date_part",
        "time_part",
        "month_slash",
        "day_slash",
        "day_dash",
        "month_dash",
        "prev_seconds",
        "year_long",
        "month_long",
        "day_long",
        "hour_long",
        "minute_long"
    )
    print("=" * 50)
    print(f"Total Columns: {len(df.columns)}")
    print(f"Final Rows: {df.count():,}")
    print("=" * 50)

    return df
