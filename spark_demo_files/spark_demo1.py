from pyspark.conf import SparkConf
from pyspark.context import SparkContext

conf = SparkConf().setMaster('local[2]').setAppName('spark_demo1')
sc = SparkContext(conf=conf)
data = [1,2,3,4,5]
distData = sc.parallelize(data,5)
print(distData.collect())
sum = distData.reduce(lambda a,b:a+b)
print(sum)
print(distData.map(lambda x:x+1).collect())
print(distData.filter(lambda x:x>3).collect())
print(distData.groupBy(lambda x:x%2==0).collect())
sc.stop()

