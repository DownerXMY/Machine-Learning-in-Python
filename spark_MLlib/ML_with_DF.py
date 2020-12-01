from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd

def data_propossessing(sc):
    hour_data = sc.textFile('/usr/local/Cellar/apache-spark/3.0.0/libexec/sources/hour.csv')
    hour_RDD = hour_data.map(lambda tp:tp.split(','))
    information_ls = [num for num in range(17)]
    for item in [0,1,3,14,15]:
        information_ls.remove(item)
    hour_RDD_information = hour_RDD.map(lambda tp:[tp[item] for item in information_ls])
    header = hour_RDD_information.first()
    hour_data_RDD = hour_RDD_information.filter(lambda line:line != header)
    print(header)
    print(hour_data_RDD.take(3))

    SchemaString = ''
    for item in header:
        SchemaString = SchemaString + item + ' '
    SchemaString = SchemaString[:-2]
    # print(SchemaString)
    fields = [StructField(field_name, StringType(), True) for field_name in SchemaString.split(' ')]
    Schema = StructType(fields)
    get_DF = spark.createDataFrame(hour_data_RDD, Schema)
    get_DF.show()
    get_DF.createOrReplaceTempView('bicycle')
    SQL_table = spark.sql(('SELECT * FROM bicycle WHERE season == 1'))
    SQL_table.show()
    # In order to accelerate while loading bid data,
    # Since sometimes we merely want to have a simple look at the data structure
    # Add the LIMIT sentence:
    SQL_table1 = spark.sql(('SELECT * FROM bicycle WHERE season == 1 LIMIT 5'))
    SQL_table1.show()
    return get_DF

def some_basic_DF_manipulation(DF):
    print('Filter:')
    summer_holiday_DF = DF.filter((DF.season==2) & (DF.holiday==1))
    summer_holiday_DF.show()
    # Note that here we can only apply '&' but can not be 'and'.
    print('Sort in SQL:')
    summer_holiday_DF.createOrReplaceTempView('summer_holiday')
    SQL_summer_holiday = spark.sql(('SELECT * FROM summer_holiday ORDER BY cn'))
    SQL_summer_holiday.show()
    # If one want to sort in decreasing, add 'DESC' after 'cn'.
    print('Sort in DF:')
    summer_holiday_ordered_DF = summer_holiday_DF.orderBy('cn')
    summer_holiday_ordered_DF.show()

def DF_from_spark_to_pandas(DF):
    DF_in_pandas = DF.toPandas()
    print(type(DF_in_pandas))
    print(DF_in_pandas.loc[:,['season','holiday']])

if __name__ == '__main__':
    spark = SparkSession \
            .builder \
            .appName('ML_with_DF') \
            .getOrCreate()
    sc = spark.sparkContext
    print('An important work is to compare ML efficiency between RDD and DF')
    hour_data_DF = data_propossessing(sc=sc)
    print('One may wonder that whether the ML work can be done with DF?')

    some_basic_DF_manipulation(DF=hour_data_DF)
    print('One question have been in our mind for a long time:')
    print('What is the relationship between DataFrames in spark and pandas?')
    DF_from_spark_to_pandas(hour_data_DF)
    print('In this file, we only introduce and review the basic points in DF and SQL',
          'The ML work based on DataFrame will be introduce in ML_with_DF2.py')