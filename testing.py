import logging
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.utils import AnalysisException

# ==========================================
# 0. SPARK SESSION + LOGGING SETUP
# ==========================================
spark = SparkSession.builder \
    .appName("CustomerDemographicsETL") \
    .getOrCreate()

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

    # FIX: Aligned schema2 column order to match schema1 (id, name, age)
    # Previously schema2 was (name, age, id) which caused a positional
    # STRING->INTEGER cast failure when union() was called in the Gold layer.
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

    # FIX: Updated row tuples to supply values in the new (id, name, age) order
    # to match the corrected schema2 definition above.
    # Previously rows were ordered as (name, age, id) e.g. ("Dave", 22, 4).
    df2_bronze = spark.createDataFrame([
        (4, "Dave", 22),
        (5, "BotAccount", -5),
        (6, "Eve", 28)
    ], schema=schema2)

    logger.info("Bronze layer schemas:")
    logger.info("df1_bronze schema: %s", df1_bronze.schema.simpleString())
    logger.info("df2_bronze schema: %s", df2_bronze.schema.simpleString())

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering noisy data for Silver layer...")
    df1_silver = df1_bronze.filter("age >= 0")
    df2_silver = df2_bronze.filter("age >= 0")

    # FIX: Added a pre-union schema compatibility guard.
    # This surfaces schema drift as a clear AssertionError rather than
    # a cryptic SparkNumberFormatException deep in the execution plan.
    assert set(df1_silver.columns) == set(df2_silver.columns), (
        f"Schema mismatch between Silver DataFrames — "
        f"df1_silver columns: {df1_silver.columns} vs "
        f"df2_silver columns: {df2_silver.columns}"
    )
    logger.info("Silver layer schema compatibility check passed.")

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # FIX: Replaced union() with unionByName() so that columns are aligned
    # by name rather than by position. This is the most defensive approach
    # and prevents this entire class of positional-union schema bugs from
    # recurring if column ordering diverges again in a future schema change.
    df_gold = df1_silver.unionByName(df2_silver)

    logger.info("Pipeline completed successfully.")
    df_gold.show()

except AssertionError as ae:
    logger.error(
        "Pipeline aborted — schema compatibility check failed. Details: %s", str(ae)
    )
    raise
except AnalysisException as ae:
    logger.error(
        "Pipeline failed with a Spark AnalysisException (likely a schema or column "
        "resolution error). Check column names and types across all DataFrames. "
        "Details: %s", str(ae)
    )
    raise
except Exception as e:
    error_msg = str(e)
    if "CAST_INVALID_INPUT" in error_msg or "SparkNumberFormatException" in error_msg:
        logger.error(
            "Pipeline failed with a CAST_INVALID_INPUT error (SQLSTATE 22018). "
            "A STRING value could not be cast to a numeric type — this typically "
            "indicates a column-order mismatch in a union() call. "
            "Details: %s", error_msg
        )
    else:
        logger.error(
            "Pipeline failed during execution. Error details: %s", error_msg
        )
    raise
