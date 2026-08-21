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
    # Canonical column order: [id, name, age] — used as the authoritative field ordering
    # for all DataFrames in this pipeline. Both schemas now match this order so that
    # positional union() calls are safe, and unionByName() remains the enforced call
    # site to guard against any future schema drift.
    schema1 = StructType([
        StructField("id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("age", IntegerType(), True)
    ])

    # schema2 corrected to use canonical column order [id, name, age] to match schema1.
    # Previously declared as [name, age, id] which, combined with positional union(),
    # caused the CAST_INVALID_INPUT / SparkNumberFormatException at Gold layer union time.
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

    # df2_bronze data tuples reordered to match the corrected schema2 field order:
    # previously (name, age, id) tuples; now (id, name, age) to align with schema2.
    df2_bronze = spark.createDataFrame([
        (4, "Dave", 22),
        (5, "BotAccount", -5),
        (6, "Eve", 28)
    ], schema=schema2)

    # ==========================================
    # 3. SILVER LAYER (Transformation)
    # ==========================================
    logger.info("Filtering noisy data for Silver layer..."
                " (rows with age < 0 are excluded; NULL age rows are also excluded"
                " because NULL >= 0 evaluates to NULL/false in Spark SQL)")
    # NOTE: rows where age IS NULL are implicitly excluded by this predicate.
    # If NULL age retention is required, change to: filter("age >= 0 OR age IS NULL")
    df1_silver = df1_bronze.filter("age >= 0")
    df2_silver = df2_bronze.filter("age >= 0")

    # Schema compatibility guard — fail fast with a clear message if column sets diverge
    # before the union, rather than producing a cryptic CAST_INVALID_INPUT at execution time.
    assert set(df1_silver.columns) == set(df2_silver.columns), (
        f"Column mismatch before union: {df1_silver.columns} vs {df2_silver.columns}"
    )

    # ==========================================
    # 4. GOLD LAYER (Integration)
    # ==========================================
    logger.info("Integrating Silver tables into Gold layer...")
    # FIX: replaced union() with unionByName() so Spark aligns columns by name rather
    # than by position. This eliminates the CAST_INVALID_INPUT / SparkNumberFormatException
    # that occurred when schema2's String 'name' column was positionally mapped into
    # schema1's IntegerType 'id' slot during the original union() call.
    df_gold = df1_silver.unionByName(df2_silver)

    # Post-union row count — provides an observable signal for monitoring and confirms
    # the union produced the expected output volume.
    gold_row_count = df_gold.count()
    logger.info("Gold layer row count: %d", gold_row_count)

    logger.info("Pipeline completed successfully.")
    display(df_gold)

except Exception as e:
    logger.error("Pipeline failed during execution. Error details: %s", str(e))
    raise
