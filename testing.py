import logging
import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import col

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

# Shared canonical column order used as the contract for all DataFrames in this pipeline
COLUMN_ORDER = ["id", "name", "age"]

try:
    # ==========================================
    # 1. SCHEMA DEFINITIONS
    # ==========================================
    logger.info("Defining explicit schemas...")

    # Single shared StructType so column-order divergence between sources is impossible
    CUSTOMER_SCHEMA = StructType([
        StructField("id",   IntegerType(), True),
        StructField("name", StringType(),  True),
        StructField("age",  IntegerType(), True)
    ])

    # ==========================================
    # 2. BRONZE LAYER (Extraction)
    # ==========================================
    logger.info("Extracting data into Bronze layer...")

    df1_bronze = spark.createDataFrame([
        (1, "Alice",       25),
        (2, "TestUser",   -15),
        (3, "Charlie",     30)
    ], schema=CUSTOMER_SCHEMA)

    # FIX: tuples reordered to (id, name, age) to match the corrected CUSTOMER_SCHEMA
    df2_bronze = spark.createDataFrame([
        (4, "Dave",        22),
        (5, "BotAccount",  -5),
        (6, "Eve",         28)
    ], schema=CUSTOMER_SCHEMA)

    # ==========================================
    # 3. SCHEMA COMPATIBILITY GUARD
    # ==========================================
    logger.info("Asserting schema compatibility between Bronze DataFrames...")
    assert set(df1_bronze.columns) == set(df2_bronze.columns), (
        f"Schema mismatch — df1_bronze columns {df1_bronze.columns} "
        f"differ from df2_bronze columns {df2_bronze.columns}"
    )
    df1_dtypes = dict(df1_bronze.dtypes)
    df2_dtypes = dict(df2_bronze.dtypes)
    for column in df1_bronze.columns:
        assert df1_dtypes[column] == df2_dtypes[column], (
            f"SCHEMA_ERROR: dtype mismatch on column '{column}': "
            f"df1={df1_dtypes[column]}, df2={df2_dtypes[column]}"
        )
    logger.info("Schema compatibility check passed.")

    # ==========================================
    # 4. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering noisy data for Silver layer...")
    df1_silver = df1_bronze.filter("age >= 0")
    df2_silver = df2_bronze.filter("age >= 0")

    # ==========================================
    # 5. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")

    # FIX: use unionByName() so columns are aligned by name, not by ordinal position.
    # Additionally, explicit select() enforces the canonical column contract and will
    # raise an early AnalysisException with a clear message if either upstream schema
    # drifts in the future.
    df_gold = (
        df1_silver.select(COLUMN_ORDER)
        .unionByName(df2_silver.select(COLUMN_ORDER))
    )

    # ==========================================
    # 6. GOLD LAYER DATA-QUALITY VALIDATION
    # ==========================================
    logger.info("Running post-Gold data-quality checks...")
    null_rows = df_gold.filter(
        col("id").isNull() | col("name").isNull() | col("age").isNull()
    ).count()
    assert null_rows == 0, (
        f"DATA_QUALITY_ERROR: df_gold contains {null_rows} row(s) with NULL values "
        "in mandatory columns (id, name, age)."
    )

    gold_count = df_gold.count()
    # After Silver filtering, valid rows are: Alice(25), Charlie(30), Dave(22), Eve(28) = 4
    EXPECTED_GOLD_ROWS = 4
    assert gold_count == EXPECTED_GOLD_ROWS, (
        f"DATA_QUALITY_ERROR: df_gold row count {gold_count} "
        f"does not match expected {EXPECTED_GOLD_ROWS}."
    )
    logger.info(
        "Data-quality checks passed. df_gold contains %d row(s) with no NULLs.",
        gold_count
    )

    logger.info("Pipeline completed successfully.")
    df_gold.show()

except (AssertionError, ValueError) as e:
    # Schema or data-quality contract violation — log with SCHEMA_ERROR tag for fast triage
    logger.error("SCHEMA_ERROR — Pipeline failed due to a schema or data-quality violation: %s", str(e))
    raise
except Exception as e:
    error_msg = str(e)
    if "CAST_INVALID_INPUT" in error_msg or "cannot be cast" in error_msg:
        logger.error(
            "SCHEMA_ERROR — Pipeline failed due to an invalid type cast (likely a column-order "
            "mismatch in a union). Error details: %s", error_msg
        )
    else:
        logger.error("Pipeline failed during execution. Error details: %s", error_msg)
    raise
