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

    # Fixed: reordered schema2 column definitions to match schema1's positional
    # order (id INT, name STRING, age INT) so that union() maps columns correctly
    # by position. Previously schema2 was (name STRING, age INT, id INT) which
    # caused Spark to cast STRING name values into INT id positions at execution
    # time, raising SparkNumberFormatException (CAST_INVALID_INPUT, SQLSTATE 22018).
    schema2 = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True)
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

    # Fixed: reordered inline data tuples for df2_bronze to match the corrected
    # schema2 column order (id INT, name STRING, age INT). Previously tuples were
    # in (name, age, id) order matching the old schema2, so values are now
    # reordered to (id, name, age) to align with the corrected schema.
    df2_bronze = spark.createDataFrame([
        (4, "Dave", 22),
        (5, "BotAccount", -5),
        (6, "Eve", 28)
    ], schema=schema2)

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering noisy data for Silver layer...")
    df1_silver = df1_bronze.filter("age >= 0")
    df2_silver = df2_bronze.filter("age >= 0")

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # Defensive schema compatibility assertion: fail fast at development/CI time
    # rather than at Spark execution time if schemas drift in the future.
    assert df1_silver.schema == df2_silver.schema, (
        f"Schema mismatch before union: "
        f"df1={df1_silver.schema}, df2={df2_silver.schema}"
    )

    # Fixed: replaced bare union() with unionByName() so that columns are aligned
    # by name rather than ordinal position. This eliminates the entire class of
    # positional column-order bugs and makes the Gold integration step resilient
    # to future schema reorderings in either upstream DataFrame.
    df_gold = df1_silver.unionByName(df2_silver)

    # Post-Gold row count assertion: ensures the Gold DataFrame is not silently
    # empty after the union before handing off to display() or downstream consumers.
    gold_count = df_gold.count()
    expected_count = df1_silver.count() + df2_silver.count()
    assert gold_count == expected_count, (
        f"Gold DataFrame row count mismatch: expected {expected_count}, "
        f"got {gold_count} — aborting pipeline."
    )

    logger.info("Pipeline completed successfully. Gold row count: %d", gold_count)
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
