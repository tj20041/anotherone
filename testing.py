import logging
import sys
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

# ==========================================
# 0. DATABRICKS-SAFE LOGGING SETUP
# ==========================================
logger = logging.getLogger("CustomerDemographicsETL")
logger.setLevel(logging.INFO)

# Prevent duplicate handlers if the cell is re-run
if not logger.handlers:
    stream_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

logger.propagate = False

logger.info("Initializing Customer Demographics ETL Job...")

try:
    # ==========================================
    # 1. SCHEMA DEFINITIONS
    # ==========================================
    logger.info("Defining explicit schemas...")
    schema1 = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True)
    ])

    schema2 = StructType([
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True),
        StructField("id", IntegerType(), True)
    ])

    # ==========================================
    # 2. BRONZE LAYER (Extraction)
    # ==========================================
    logger.info("Extracting data into Bronze layer...")
    df1_bronze = spark.createDataFrame([
        (1, "Alice", 25),
        (2, "TestUser", -15),
        (3, "Charlie", 30)
    ], schema=schema1)

    df2_bronze = spark.createDataFrame([
        ("Dave", 22, 4),
        ("BotAccount", -5, 5),
        ("Eve", 28, 6)
    ], schema=schema2)

    logger.info("Bronze layer row counts: df1=%d, df2=%d", df1_bronze.count(), df2_bronze.count())

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering noisy data for Silver layer...")
    df1_silver = df1_bronze.filter("age >= 0")
    df2_silver = df2_bronze.filter("age >= 0")

    logger.info("Silver layer row counts: df1=%d, df2=%d", df1_silver.count(), df2_silver.count())

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # Pre-union schema compatibility check: ensure both DataFrames share the same column names
    mismatched_cols = set(df1_silver.columns).symmetric_difference(set(df2_silver.columns))
    if mismatched_cols:
        raise ValueError(f"Cannot union — column mismatch: {mismatched_cols}")

    # Use unionByName() to align columns by name rather than position,
    # preventing cast failures caused by differing column orderings in schema1 vs schema2
    df_gold = df1_silver.unionByName(df2_silver)

    # Post-union row count validation to detect silent data loss
    expected_count = df1_silver.count() + df2_silver.count()
    actual_count = df_gold.count()
    assert actual_count == expected_count, (
        f"Gold row count ({actual_count}) does not match sum of Silver inputs "
        f"({expected_count}) — possible data loss in union step"
    )

    logger.info("Gold layer row count: %d (expected %d)", actual_count, expected_count)
    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error(
        "Pipeline failed during execution. Offending column types — df1: %s, df2: %s. Error: %s",
        df1_silver.dtypes if 'df1_silver' in dir() else 'N/A',
        df2_silver.dtypes if 'df2_silver' in dir() else 'N/A',
        str(e)
    )
    raise
