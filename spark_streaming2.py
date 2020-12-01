import sys
from pyspark.context import SparkContext
from pyspark.conf import SparkConf
from pyspark.streaming import StreamingContext

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: spark_streaming2.py <hostname> <port>',file=sys.stderr)
        sys.exit(-1)
    conf = SparkConf().setAppName('streaming_demo')
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc,10)
    # Note that the 10 is called the batch interval.
    lines = ssc.socketTextStream(hostname=sys.argv[1],port=int(sys.argv[2]))
    counts = lines.flatMap(lambda line:line.split(' '))\
        .map(lambda word:(word,1))\
        .reduceByKey(lambda a,b:a+b)
    counts.pprint()

    ssc.start()
    ssc.awaitTermination()