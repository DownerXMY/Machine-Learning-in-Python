# 和RDD不同的是，RDD的入口点是SpartContext，但是DataFrame不是，是下面的SparkSession
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.sql.types import StructField,StringType,StructType
# the above two import can be replaced by from pyspark.sql.types import *

def My_func1(spark):
    df = spark.read.json('/usr/local/Cellar/apache-spark/3.0.0/libexec/examples/src/main/resources/people.json')
    df.show()
    df.printSchema()
    df.select('name').show()
    df.select(df['name'],df['age']+1).show()
    df.filter(df['age'] > 25).show()
    print('We can also apply the SQL matipulations:')
    df.createOrReplaceTempView('people')
    sqlDF = spark.sql(('SELECT * FROM people')) # This is kind of SQL language, which may be different from py.
    # Note that here sqlDF is also a DataFrame.
    sqlDF.show()

def RDD_to_DF(rdd):
    c_table = rdd.map(lambda x:Row(name=x[0],age=int(x[1])))
    c_dataframe = spark.createDataFrame(c_table)
    c_dataframe.createOrReplaceTempView('people')
    sqlDF1 = spark.sql(('SELECT * FROM people WHERE age >= 20 AND age <= 32'))
    sqlDF1.show()
    return sqlDF1

def DF_to_RDD(df):
    return df.rdd

def Add_Schema_to_RDD(rdd):
    table = rdd.map(lambda x:(x[0],x[1]))
    schemaString = 'name age'
    fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split(' ')]
    schema = StructType(fields)
    get_DF = spark.createDataFrame(table,schema)
    get_DF.show()
    get_DF.createOrReplaceTempView('people')
    sqlDF2 = spark.sql(('SELECT * FROM people WHERE length(name) <= 5'))
    sqlDF2.show()

if __name__ == '__main__':
    spark = SparkSession \
    .builder \
    .appName('Spark SQL simple_example') \
    .getOrCreate()

    print('basic manipulations in DF:')
    My_func1(spark)

    sc = spark.sparkContext
    data = sc.textFile('/Users/mingyuexu/PycharmProjects/demo/learning/spark_demo_files/spark_demo3.txt')
    data_rdd = data.map(lambda x:x.split(','))
    # print(data_rdd.collect())

    print('Transfer RDD to DataFrame:')
    print('Row + createDataFrame:')
    RDD_to_DF(data_rdd)
    print('DataFrame to RDD:')
    get_rdd = DF_to_RDD(RDD_to_DF(data_rdd))
    print(get_rdd.collect())
    print('note that the above rdd is somewhat hard to understand!')
    print('Take care that one is able to get information from the column index of the DataFrame')
    name_rdd = get_rdd.map(lambda x:x.name)
    age_rdd = get_rdd.map(lambda x:x.age)
    print(name_rdd.collect(),age_rdd.collect())

    print('We have more than one way to transfer RDD to DataFrame')
    print('Add Schema to RDD:')
    Add_Schema_to_RDD(data_rdd)

    spark.stop()