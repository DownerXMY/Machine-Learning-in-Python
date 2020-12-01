import sys
sys.path.append('/Users/mingyuexu/PycharmProjects/demo/learning/spark_MLlib')
from pyspark.conf import SparkConf
from pyspark.context import SparkContext

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.evaluation import BinaryClassificationMetrics

from learning.spark_MLlib.spark_Dtree import train_data_propossesssing
from learning.spark_MLlib.spark_Dtree import establish_form
from learning.spark_MLlib.spark_Dtree import split_data

import numpy as np

def data_standardlize(rdd):
    label_data = rdd.map(lambda p:p.label)
    feature_data = rdd.map(lambda p:p.features)
    stdScaler = StandardScaler(withMean=True,withStd=True).fit(feature_data)
    Scaler_feature_RDD = stdScaler.transform(feature_data)
    labelpoint_data = label_data.zip(Scaler_feature_RDD)
    labelpoint_RDD = labelpoint_data.map(lambda pair:LabeledPoint(pair[0],pair[1]))
    return labelpoint_RDD

def model_pred_accu(trainData,testData,validationData):
    model = LogisticRegressionWithLBFGS.train(trainData,iterations=1000,validateData=validationData,numClasses=2)
    test_feature = testData.map(lambda p:p.feature)
    pred_result = model.predict(test_feature)
    flattens = pred_result.zip(testData.map(lambda p: p.label))
    metrics = BinaryClassificationMetrics(flattens)
    print('AUC=', metrics.areaUnderROC)

if __name__ == '__main__':
    conf = SparkConf().setAppName('spark_LogisticR')
    sc = SparkContext(conf=conf)
    raw_data = sc.textFile('/Users/mingyuexu/PycharmProjects/demo/learning/spark_demo_files/sources/train.tsv')
    data_propossessed = train_data_propossesssing(raw_data,sc=sc)
    data_labeled = establish_form(data_propossessed)
    data_standard_RDD = data_standardlize(data_labeled)
    trainData,validationData,testData = split_data(data_standard_RDD)
    model_pred_accu(trainData=trainData,testData=testData,validationData=validationData)
