# Script demonstrating correct PySpark UDF and native Spark SQL column casting on Databricks
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

spark = SparkSession.builder.getOrCreate()

# Create a DataFrame
df = spark.createDataFrame([(1, "A"), (2, "B")], ["id", "letter"])

# FIX 1 (direct): Replace the broken UDF that called the Java `.toString()` method
# on a native Python int (which has no such attribute) with a correct Python UDF
# that uses the built-in str() conversion and guards against null values.
def bad_udf(x):
    # str() converts the native Python int to a string — no Java method call needed.
    # The None guard prevents a TypeError if the column contains null values.
    return str(x) if x is not None else None

# FIX 2 (related): Register the UDF with an explicit StringType return type.
# Without this, Spark silently defaults to StringType but cannot validate it at
# registration time, meaning type mismatches propagate silently through the DAG
# and only surface at execution time. Explicit return types are required best
# practice on Databricks.
bad_udf_spark = udf(bad_udf, StringType())
df.withColumn("bad", bad_udf_spark("id")).show()

# FIX 3 (recommended alternative): Prefer native Spark SQL expressions over Python
# UDFs for simple type coercions. col('id').cast('string') runs entirely inside the
# JVM via Photon/Tungsten, requires no Python serialisation round-trip, and avoids
# the entire class of Py4J/Python-UDF runtime errors. On Databricks with Photon
# enabled this path is also Photon-eligible, giving further performance benefits.
df.withColumn("bad", col("id").cast("string")).show()
