# databricks_script_with_error.py - This WILL FAIL

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create Spark session
spark = SparkSession.builder.appName("ErrorApp").getOrCreate()

# Create sample data
print("Creating sample data...")
data = [
    (1, "john", 50000),
    (2, "jane", 60000),
    (3, "bob", 70000),
    (4, "alice", 55000)
]

# Create DataFrame
df = spark.createDataFrame(data, ["id", "name", "salary"])
print(f"Created DataFrame with {df.count()} rows")

# Show original data
print("\nOriginal Data:")
df.show()

# THIS WILL CAUSE AN ERROR - Division by zero
print("\n⚠️ About to cause an error...")
df_error = df.withColumn("error_col", col("salary") / 0)

# This line never runs because the above fails
print("This will never print")
df_error.show()
