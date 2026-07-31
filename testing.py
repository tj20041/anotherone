# databricks_script.py - Working Databricks Script

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, upper

# Create Spark session
spark = SparkSession.builder.appName("SimpleApp").getOrCreate()

# Create sample data
print("Creating sample data...")
data = [
    (1, "john", "sales", 50000),
    (2, "jane", "marketing", 60000),
    (3, "bob", "it", 70000),
    (4, "alice", "hr", 55000)
]

# Create DataFrame
df = spark.createDataFrame(data, ["id", "name", "department", "salary"])
print(f"Created DataFrame with {df.count()} rows")

# Show original data
print("\nOriginal Data:")
df.show()

# Simple transformations
print("\nTransforming data...")
df_transformed = df.withColumn("name_upper", upper(col("name")))
df_transformed = df_transformed.withColumn("salary_bonus", col("salary") * 0.10)

# Show transformed data
print("\nTransformed Data:")
df_transformed.show()

# Calculate summary statistics
print("\nSummary Statistics:")
df.select("salary").describe().show()

# Save as temp view for SQL queries
df.createOrReplaceTempView("employees")

# Run SQL query
print("\nSQL Query Results:")
sql_result = spark.sql("SELECT department, AVG(salary) as avg_salary FROM employees GROUP BY department")
sql_result.show()

print("\n✅ Script completed successfully!")
