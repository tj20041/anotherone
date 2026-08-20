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

    # Pre-union schema compatibility assertion: verify both DataFrames share
    # the same column names (order-independent) and dtypes before unioning,
    # so any future schema drift is caught early with a clear message.
    assert set(df1_silver.columns) == set(df2_silver.columns), (
        f"Column name mismatch before union: "
        f"{df1_silver.columns} vs {df2_silver.columns}"
    )
    assert set(df1_silver.dtypes) == set(df2_silver.dtypes), (
        f"Column dtype mismatch before union: "
        f"{df1_silver.dtypes} vs {df2_silver.dtypes}"
    )

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    # Use unionByName() instead of union() to align columns by name rather
    # than by position.  schema1 is (id, name, age) and schema2 is
    # (name, age, id) — a positional union() would map df2_silver's STRING
    # 'name' column into df1_silver's INTEGER 'id' slot, causing
    # SparkNumberFormatException [CAST_INVALID_INPUT] SQLSTATE 22018.
    logger.info(
        "Integrating Silver tables into Gold layer using unionByName "
        "to enforce column alignment..."
    )
    df_gold = df1_silver.unionByName(df2_silver)

    # Gold-layer row-count validation: the unified result must contain exactly
    # as many rows as the two Silver DataFrames combined.
    expected_count = df1_silver.count() + df2_silver.count()
    actual_count = df_gold.count()
    assert actual_count == expected_count, (
        f"Gold layer row count mismatch: expected {expected_count}, "
        f"got {actual_count}"
    )
    logger.info(
        "Gold layer row-count validation passed: %d rows (df1_silver=%d, "
        "df2_silver=%d).",
        actual_count,
        df1_silver.count(),
        df2_silver.count(),
    )

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
