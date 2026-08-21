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

    # Reordered schema2 columns to match schema1's positional order (id, name, age)
    # as a defensive measure so both union() and unionByName() produce correct results.
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

    # Reordered data tuples to match the corrected schema2 column order (id, name, age)
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

    # Schema compatibility assertion: catch column-name mismatches early with a
    # clear, actionable error rather than a cryptic CAST_INVALID_INPUT from Spark.
    assert set(df1_silver.columns) == set(df2_silver.columns), (
        f"Schema mismatch before union: df1={df1_silver.columns}, df2={df2_silver.columns}"
    )

    # Use unionByName() instead of union() so columns are matched by name rather
    # than position, preventing STRING-to-BIGINT cast failures when column order
    # differs between the two DataFrames.
    df_gold = df1_silver.unionByName(df2_silver)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    # Log both DataFrame schemas at failure time to dramatically reduce triage
    # time if a positional or type mismatch recurs after future schema changes.
    try:
        logger.error("df1_silver schema: %s", df1_silver.schema.simpleString())
        logger.error("df2_silver schema: %s", df2_silver.schema.simpleString())
    except Exception:
        logger.error("Could not retrieve DataFrame schemas for diagnostic logging.")
    raise
