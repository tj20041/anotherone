import logging
import sys
from pyspark.sql import SparkException
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
    # Canonical column order for ALL source schemas: [id (INT), name (STRING), age (INT)]
    # Both schema1 and schema2 must follow this order so that positional union()
    # calls made by future developers remain safe. unionByName() is used below as
    # a belt-and-suspenders guard, but consistent ordering is the primary contract.
    # ==========================================
    logger.info("Defining explicit schemas...")

    # schema1: canonical order [id, name, age]
    schema1 = StructType([
        StructField("id",   IntegerType(), True),
        StructField("name", StringType(),  True),
        StructField("age",  IntegerType(), True)
    ])

    # schema2: corrected to match canonical order [id, name, age]
    # Previously this was [name, age, id] which caused STRING->BIGINT cast failures
    # when union() aligned columns positionally across the two DataFrames.
    schema2 = StructType([
        StructField("id",   IntegerType(), True),
        StructField("name", StringType(),  True),
        StructField("age",  IntegerType(), True)
    ])

    # ==========================================
    # 2. BRONZE LAYER (Extraction)
    # ==========================================
    logger.info("Extracting data into Bronze layer...")
    df1_bronze = spark.createDataFrame([
        (1, "Alice",   25),
        (2, "TestUser", -15),
        (3, "Charlie",  30)
    ], schema=schema1)

    # Tuples reordered to (id, name, age) to match the corrected schema2 field order.
    # Previously tuples were (name, age, id) e.g. ("Dave", 22, 4) which matched the
    # old (wrong) schema2 ordering; they must now be (4, "Dave", 22) etc.
    df2_bronze = spark.createDataFrame([
        (4, "Dave",       22),
        (5, "BotAccount", -5),
        (6, "Eve",        28)
    ], schema=schema2)

    # Defensive schema alignment guard: assert both Bronze frames expose identical
    # column names before any transformation proceeds, catching divergence early.
    assert set(df1_bronze.columns) == set(df2_bronze.columns), (
        f"Bronze schema mismatch: df1_bronze columns {df1_bronze.columns} "
        f"do not match df2_bronze columns {df2_bronze.columns}"
    )
    logger.info("Bronze schema alignment check passed.")

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering noisy data for Silver layer...")
    df1_silver = df1_bronze.filter("age >= 0")
    df2_silver = df2_bronze.filter("age >= 0")

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # Explicit select enforces canonical column order [id, name, age] on both
    # Silver DataFrames before the union, providing a deterministic schema
    # contract and guarding against silent regressions if upstream schemas change.
    # unionByName() is used instead of union() so that columns are aligned by
    # name rather than position — eliminating the CAST_INVALID_INPUT error that
    # occurred when positional union() mapped df2_silver's STRING 'name' column
    # onto df1_silver's INT 'id' column.
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")
    df1_silver = df1_silver.select("id", "name", "age")
    df2_silver = df2_silver.select("id", "name", "age")

    df_gold = df1_silver.unionByName(df2_silver)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    error_str = str(e)
    # Emit a specific, actionable alert when a schema / cast mismatch is detected
    # in a union operation so that on-call engineers can identify the problem
    # without needing to read the full Spark stack trace.
    if "CAST_INVALID_INPUT" in error_str or "SQLSTATE: 22018" in error_str:
        logger.error(
            "Schema mismatch detected in union operation — check column ordering "
            "and types between source DataFrames. Error details: %s", error_str
        )
    else:
        logger.error("Pipeline failed during execution. Error details: %s", error_str)
    raise
