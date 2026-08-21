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

    # Fixed: reordered schema2 columns to match schema1's column order (id, name, age)
    # Previously schema2 was defined as (name, age, id) which caused positional misalignment
    # during union(), resulting in SparkNumberFormatException (CAST_INVALID_INPUT / SQLSTATE 22018)
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

    # Fixed: reordered tuples to match the corrected schema2 column order (id, name, age)
    # Previously tuples were ordered as (name, age, id) e.g. ("Dave", 22, 4)
    # Now correctly ordered as (id, name, age) e.g. (4, "Dave", 22)
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

    # Pre-union schema compatibility assertion: catch any future schema drift early
    # with a clear, actionable error rather than a silent positional misalignment
    if df1_silver.columns != df2_silver.columns:
        raise ValueError(
            f"Schema mismatch before union: "
            f"df1_silver columns {df1_silver.columns} "
            f"do not match df2_silver columns {df2_silver.columns}. "
            f"Resolve column name or order differences before proceeding."
        )

    if df1_silver.dtypes != df2_silver.dtypes:
        raise ValueError(
            f"Type mismatch before union: "
            f"df1_silver dtypes {df1_silver.dtypes} "
            f"do not match df2_silver dtypes {df2_silver.dtypes}. "
            f"Resolve column type differences before proceeding."
        )

    # Using unionByName() for robust name-based column alignment, making the union
    # resilient to any future schema column-order discrepancies between source DataFrames
    df_gold = df1_silver.unionByName(df2_silver)

    # Post-union row count assertion to detect silent data loss
    df1_count = df1_silver.count()
    df2_count = df2_silver.count()
    gold_count = df_gold.count()
    logger.info(
        "Row counts — df1_silver: %d, df2_silver: %d, df_gold: %d",
        df1_count, df2_count, gold_count
    )
    if gold_count < (df1_count + df2_count):
        raise ValueError(
            f"Gold layer row count {gold_count} is less than the sum of Silver inputs "
            f"({df1_count} + {df2_count} = {df1_count + df2_count}). "
            f"The union may have silently dropped rows."
        )

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
