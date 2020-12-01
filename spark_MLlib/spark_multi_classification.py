from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import MulticlassMetrics

def data_visual_DF(sc):
    data = sc.textFile('/usr/local/Cellar/apache-spark/3.0.0/libexec/sources/covtype.data')
    raw_RDD = data.map(lambda x:x.split(','))
    schemaString = ''
    for item in ['features'+str(item) for item in range(10)]:
        schemaString = schemaString + item + ' '
    for item in ['W_type'+str(item) for item in range(4)]:
        schemaString = schemaString + item + ' '
    for item in ['S_type'+str(item) for item in range(40)]:
        schemaString = schemaString + item + ' '
    schemaString = schemaString + 'label'

    fields = [StructField(field_name, StringType(), True) for field_name in schemaString.split(' ')]
    schema = StructType(fields)
    get_DF = spark.createDataFrame(raw_RDD, schema)
    get_DF.show()
    get_DF.createOrReplaceTempView('covtype')
    sqlDF2 = spark.sql(('SELECT * FROM covtype WHERE length(W_type1) == 1'))
    sqlDF2.show()

def data_propossessing(sc):
    data = sc.textFile('/usr/local/Cellar/apache-spark/3.0.0/libexec/sources/covtype.data')
    raw_data = data.map(lambda x:x.split(','))
    # print(raw_data.take(5))
    data_labeled = raw_data.map(lambda tp:LabeledPoint(
        label=float(int(tp[-1])-1),
        features=[float(item) for item in tp[0:len(tp)-1]]
    ))
    print(data_labeled.take(5))
    return data_labeled

def data_spliting(rdd):
    trainData,testData = rdd.randomSplit([8,2])
    print(trainData.count(),testData.count())
    return trainData,testData

def modeling_pred_accu(trainData,testData):
    model = DecisionTree.trainClassifier(data=trainData,numClasses=7,
                                         categoricalFeaturesInfo={},
                                         impurity='entropy')
    test_features = testData.map(lambda pair:pair.features)
    pred_result = model.predict(test_features)
    accuracy_pair = pred_result.zip(testData.map(lambda pair:pair.label))
    metrics = MulticlassMetrics(accuracy_pair)
    accuracy = metrics.accuracy
    return accuracy

if __name__ == '__main__':
    conf = SparkConf().setAppName('spark_multi_classification')
    sc = SparkContext(conf=conf)
    spark = SparkSession \
            .builder \
            .appName('spark_multi_classification') \
            .getOrCreate()
    sc1 = spark.sparkContext
    # data_visual_DF(sc=sc1)
    data_labeled = data_propossessing(sc=sc)
    trainData,testData = data_spliting(data_labeled)
    accuracy = modeling_pred_accu(trainData=trainData,testData=testData)
    print(accuracy)