# Script that will cause Py4JJavaError
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf

spark = SparkSession.builder.getOrCreate()

# Create a DataFrame
df = spark.createDataFrame([(1, "A"), (2, "B")], ["id", "letter"])

# THE ERROR: UDF that references Java object incorrectly
def bad_udf(x):
    # This will cause a Py4J error because it tries to access Java object
    return x.toString()  # Invalid operation on int

bad_udf_spark = udf(bad_udf)
df.withColumn("bad", bad_udf_spark("id")).show()
