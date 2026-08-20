import logging
import sys
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import col

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

    # FIX: Standardised schema2 column ordering to match schema1 (id, name, age)
    # Previously was (name, age, id) which caused positional misalignment in union()
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

    # FIX: Reordered df2_bronze tuples from (name, age, id) to (id, name, age)
    # to match the corrected schema2 field ordering
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

    # Log schemas at INFO level so any future drift is visible in pipeline logs
    logger.info("df1_silver schema: %s", df1_silver.schema.simpleString())
    logger.info("df2_silver schema: %s", df2_silver.schema.simpleString())

    # FIX: Added schema compatibility assertion before union to catch mismatches
    # at development time with a clear, readable error message
    assert df1_silver.columns == df2_silver.columns, (
        f"Column mismatch before union: {df1_silver.columns} vs {df2_silver.columns}"
    )

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # FIX: Replaced union() with unionByName() so columns are aligned by name
    # rather than position, eliminating the CAST_INVALID_INPUT / SQLSTATE 22018
    # SparkNumberFormatException caused by positional misalignment between
    # schema1 (id, name, age) and the original schema2 (name, age, id)
    df_gold = df1_silver.unionByName(df2_silver)

    # FIX: Added Gold-layer data quality check to verify 'id' column contains
    # only non-null integer-compatible values before any downstream action
    bad_id_count = df_gold.filter(
        col("id").isNull() | (~col("id").cast("string").rlike("^-?[0-9]+$"))
    ).count()
    assert bad_id_count == 0, (
        f"Data quality failure: {bad_id_count} row(s) in df_gold have a null or "
        "non-integer 'id' value. Check source schemas for type misalignment."
    )

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
