from flask import Flask
import random
import pandas as pd
from pyspark.sql import SparkSession

app = Flask(__name__)

# Create a SparkSession
spark = SparkSession.builder.appName("RandomRecordAPI").getOrCreate()

# Define the Delta table path
delta_table_path = "/mnt/raw/src_data/iqubeug_charge_data/"

# Read data from the Delta table
delta_df = spark.read.format("delta").load(delta_table_path)

# Define the number of random rows you want to store (between 10 and 20)
num_random_rows = random.randint(10, 20)

# Use the sample method to select a random subset of rows
sampled_df = delta_df.sample(fraction=num_random_rows / delta_df.count(), seed=42)

# Convert the sampled DataFrame to Pandas
pandas_df = sampled_df.toPandas()

@app.route('/get_random_record', methods=['GET'])
def get_random_record():
    try:
        random_index = random.randint(0, pandas_df.shape[0] - 1)
        random_record = pandas_df.iloc[random_index:random_index + 1]

        # Convert the record to an HTML table
        html_table = random_record.to_html(index=False)
        return html_table
    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
