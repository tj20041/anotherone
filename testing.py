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

    # Schema2 column order standardised to match schema1 (id, name, age)
    # to ensure consistent positional alignment and prevent future union bugs.
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

    # Row tuples updated to match standardised schema2 column order (id, name, age).
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
    # Changed from union() to unionByName() to merge columns by name rather
    # than by position, preventing STRING-to-BIGINT cast failures caused by
    # differing column declaration orders across schemas.
    df_gold = df1_silver.unionByName(df2_silver)

    # Post-union schema assertion to catch regressions from future schema drift.
    expected_columns = ["id", "name", "age"]
    expected_schema = schema1
    if df_gold.columns != expected_columns:
        raise AssertionError(
            "Gold layer schema mismatch — unexpected column order or names. "
            "Expected: %s, Got: %s" % (expected_columns, df_gold.columns)
        )
    if df_gold.schema != expected_schema:
        raise AssertionError(
            "Gold layer schema mismatch — unexpected column types. "
            "Expected schema: %s, Got schema: %s" % (expected_schema, df_gold.schema)
        )
    logger.info("Gold layer schema assertion passed. Columns: %s", df_gold.columns)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
