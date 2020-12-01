if __name__ == '__main__':
    from pyspark.context import SparkContext
    from pyspark.conf import SparkConf
    # from pyspark import StorageLevel

    # persist and cache
    conf = SparkConf().setAppName('spark_demo2')
    sc = SparkContext(conf=conf)


    # data = [1,2,3,4,5]
    # distData = sc.parallelize(data)
    # distData.persist(storageLevel=StorageLevel.MEMORY_ONLY_2)

    # rdd.cache()
    # rdd.persist(StorageLevel)
    # rdd.unpersist()
    # StorageLevel.DISK_ONLY = StorageLevel(True, False, False, False)
    # StorageLevel.DISK_ONLY_2 = StorageLevel(True, False, False, False, 2)
    # StorageLevel.MEMORY_ONLY = StorageLevel(False, True, False, False)
    # StorageLevel.MEMORY_ONLY_2 = StorageLevel(False, True, False, False, 2)
    # StorageLevel.MEMORY_AND_DISK = StorageLevel(True, True, False, False)
    # StorageLevel.MEMORY_AND_DISK_2 = StorageLevel(True, True, False, False, 2)
    # StorageLevel.OFF_HEAP = StorageLevel(True, True, True, False, 1)

    def My_func():
        result = sc.textFile('file:///Users/mingyuexu/PycharmProjects/demo/learning/spark_demo_files/spark_demo2.txt') \
            .flatMap(lambda x: x.split('\t')) \
            .map(lambda x: (x, 1)) \
            .reduceByKey(lambda a, b: a + b) \
            .collect()
        print(result)


    My_func()
    sc.stop()
