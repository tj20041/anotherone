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

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering noisy data for Silver layer...")
    df1_silver = df1_bronze.filter("age >= 0")
    df2_silver = df2_bronze.filter("age >= 0")

    # Log Silver layer row counts for data quality audit trail
    silver1_count = df1_silver.count()
    silver2_count = df2_silver.count()
    logger.info("Silver layer row counts — df1_silver: %d, df2_silver: %d", silver1_count, silver2_count)

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # Guard: assert both DataFrames share the same set of column names before
    # unioning. If schemas diverge in a future change this raises a clear,
    # descriptive error at validation time rather than a cryptic
    # CAST_INVALID_INPUT at Spark execution time.
    assert set(df1_silver.columns) == set(df2_silver.columns), (
        f"Schema mismatch before union: {df1_silver.columns} vs {df2_silver.columns}"
    )

    # Fix: use unionByName() instead of union() so that columns are aligned by
    # NAME rather than by position. union() was mapping df2_silver's StringType
    # 'name' column into df1_silver's IntegerType 'id' slot (positional slot 0),
    # causing Spark to attempt a STRING -> BIGINT cast and raising
    # SparkNumberFormatException CAST_INVALID_INPUT (SQLSTATE 22018).
    df_gold = df1_silver.unionByName(df2_silver)

    # Guard: confirm the Gold layer was produced correctly before surfacing the
    # result downstream. Catches silent data loss from overly aggressive filters.
    gold_count = df_gold.count()
    assert gold_count > 0, "Gold layer is empty after union — check Silver filters"
    logger.info("Gold layer row count: %d", gold_count)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    # exc_info=True appends the full traceback to the log entry, which is
    # captured in the Databricks driver log and cluster event log for faster
    # diagnosis.
    logger.error("Pipeline failed during execution. Error details: %s", str(e), exc_info=True)
    raise
