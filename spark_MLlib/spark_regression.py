from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import RegressionMetrics
import pandas as pd

def pd_read():
    df_hour = pd.read_csv('/usr/local/Cellar/apache-spark/3.0.0/libexec/sources/hour.csv')
    print(df_hour.loc[0:4])
    return df_hour.keys()

def data_propossessing(sc):
    day_data = sc.textFile('/usr/local/Cellar/apache-spark/3.0.0/libexec/sources/day.csv')
    hour_data = sc.textFile('/usr/local/Cellar/apache-spark/3.0.0/libexec/sources/hour.csv')
    R_hour_RDD = hour_data.map(lambda x:x.split(','))
    header = R_hour_RDD.first()
    R_hour_RDD_cor = R_hour_RDD.filter(lambda line:line != header)
    print(R_hour_RDD_cor.take(3))
    exist_ls = [num for num in range(17)]
    for item in [0,1,3,14,15]:
        exist_ls.remove(item)
    data_stage1 = R_hour_RDD_cor.map(lambda tp:[tp[item] for item in exist_ls])
    print('informaiton data remains:',data_stage1.take(3))
    data_stage2 = data_stage1.map(lambda tp:LabeledPoint(
        label=float(tp[-1]),
        features=[float(item) for item in tp[0:len(tp)-1]]
    ))
    print('labeled data:',data_stage2.take(3))
    return data_stage2

def data_spliting(rdd):
    trainData,testData = rdd.randomSplit([8,2])
    print(trainData.count(),testData.count())
    return trainData,testData

def model_and_prediction(trainData,testData):
    maxDepthlist = [3,5,10,15,20,25]
    maxBinslist = [3,5,10,50,100,200]
    model_selection_dict = {}
    for num_i in range(0,6,1):
        for num_j in range(0,6,1):
            model_choice = DecisionTree.trainRegressor(data=trainData,
                                                       categoricalFeaturesInfo={},
                                                       impurity='variance',
                                                       maxDepth=maxDepthlist[num_i],
                                                       maxBins=maxBinslist[num_j])
            test_features = testData.map(lambda tp:tp.features)
            pred_result = model_choice.predict(test_features)
            flatten = pred_result.zip(testData.map(lambda tp:tp.label))
            metrics = RegressionMetrics(flatten)
            RMSE = metrics.rootMeanSquaredError
            print(f'model based on maxDepth={maxDepthlist[num_i]} and maxBins={maxBinslist[num_j]}:',RMSE)
            model_selection_dict[(num_i,num_j)] = RMSE
    best_result = 200
    for pair in model_selection_dict.items():
        if pair[1] < best_result:
            best_result = pair[1]
    _num_i = list(model_selection_dict.keys())[list(model_selection_dict.values()).index(best_result)][0]
    _num_j = list(model_selection_dict.keys())[list(model_selection_dict.values()).index(best_result)][1]
    best_model_string = 'The best model parameters come as:' \
                        + f'maxDepth={maxDepthlist[_num_i]}, maxBins={maxBinslist[_num_j]} -- ' \
                        + f'MSE={best_result}'
    return best_model_string

if __name__ == '__main__':
    # spark = SparkSession \
    #         .builder \
    #         .appName('spark_regression') \
    #         .getOrCreate()
    # sc = spark.sparkContext
    conf = SparkConf().setAppName('spark_regression')
    sc1 = SparkContext(conf=conf)
    # print(pd_read())
    data_labeled = data_propossessing(sc=sc1)
    trainData, testData = data_spliting(data_labeled)
    best_model_string = model_and_prediction(trainData=trainData,testData=testData)
    print(best_model_string)