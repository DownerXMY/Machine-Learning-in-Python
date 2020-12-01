# spark streaming
from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.streaming import StreamingContext
# Note that the the StreamingContext is the main entrance of the streaming work
conf = SparkConf().setAppName('Spark Streaming')
sc = SparkContext(conf=conf)
print('The core of Spark-Streaming is called the DStream:')
Ds = StreamingContext(sc,batchDuration=1)
print('Ds is a DStream')
# Note that the following Ds is exactly the established DStream
lines = Ds.socketTextStream('localhost',9999)
print('An important thing that one must know is that:')
print('DStream is based on RDD, hence they may share similar manipulations:')
works = lines.flatMap(lambda x:x.split(' '))
# But make sure that the got works are all new DStreams, but not RDDs
works_after_mani = works.map(lambda x:(x,1)).reduceByKey(lambda x,y:x + y)
# An obvious discrepancy is that there is no API called .collect()
works_after_mani.pprint()

# Last but also important, we need to add the following sentence to broach the computation:
Ds.start()
Ds.awaitTermination() # expect the result to the terminal.

# Actually, this file has been an example in our resources, which pathed as:
# /usr/local/Cellar/apache-spark/3.0.0/libexec/examples/src/main/python/streaming/network_wordcount.py

print('All in all, we need a SparkContext to initialize the StreamingContext')