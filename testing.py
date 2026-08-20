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

    # ==========================================
    # 3b. SCHEMA VALIDATION GUARD
    # ==========================================
    assert set(df1_silver.columns) == set(df2_silver.columns), (
        f"Schema mismatch before union: df1_silver columns={df1_silver.columns} "
        f"vs df2_silver columns={df2_silver.columns}"
    )
    logger.info("Schema validation passed: both Silver DataFrames share the same column names.")

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")
    # FIX: replaced positional union() with unionByName() to align columns by name,
    # not by position. The original union() caused df2_silver.name (STRING) to be
    # placed in the df1_silver.id (INTEGER) slot, producing SparkNumberFormatException
    # CAST_INVALID_INPUT (SQLSTATE 22018) when Spark tried to cast 'Dave'/'Eve' to BIGINT.
    # allowMissingColumns=True makes the Gold integration robust to partial-schema
    # upstream additions without requiring a code change each time.
    df_gold = df1_silver.unionByName(df2_silver, allowMissingColumns=True)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    # Log both Silver DataFrame schemas to aid debugging of schema drift
    try:
        logger.error("df1_silver schema: %s", df1_silver.schema.simpleString())
    except Exception:
        logger.error("df1_silver schema unavailable (DataFrame may not have been created).")
    try:
        logger.error("df2_silver schema: %s", df2_silver.schema.simpleString())
    except Exception:
        logger.error("df2_silver schema unavailable (DataFrame may not have been created).")
    raise
