# Script that demonstrates correct PySpark column transformation on Databricks
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

spark = SparkSession.builder.getOrCreate()

# Create a DataFrame
df = spark.createDataFrame([(1, "A"), (2, "B")], ["id", "letter"])

# FIX: Replace the broken Python UDF (which called the Java/Scala-only .toString()
# method on a Python int, causing AttributeError -> Py4JJavaError at executor time)
# with a native Spark cast() expression.
#
# Using col("id").cast("string") instead of a Python UDF:
#   1. Executes entirely on the JVM via Catalyst — no Python serialisation overhead.
#   2. Avoids Py4J round-trips on Databricks, which is significantly faster at scale.
#   3. Is fully compatible with Databricks runtime schema enforcement (StringType output
#      is explicit and does not rely on Spark's default UDF return-type inference).
df.withColumn("bad", col("id").cast("string")).show()
