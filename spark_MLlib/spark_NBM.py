from pyspark.conf import SparkConf
from pyspark.context import SparkContext

from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.evaluation import BinaryClassificationMetrics
from pyspark.mllib.feature import StandardScaler

import numpy as np

def ancillary_func(ls,dict):
    OneHotFeature = [dict[ls[0]]]
    OneHotFeature.extend(ls[1:])
    return OneHotFeature

def data_propossessing(sc):
    raw_data = sc.textFile('file:///Users/mingyuexu/PycharmProjects/demo/learning/spark_demo_files/sources/train.tsv')
    header = raw_data.first()
    data_stage1 = raw_data.filter(lambda line:line != header)
    data_stage2 = data_stage1.map(lambda tp:tp.split(','))
    data_stage3 = data_stage2.map(lambda tp:tp[-1])
    data_stage4 = data_stage3.map(lambda item:item.split('\t'))
    data_stage5 = data_stage4.map(lambda tp:tp[1:])
    data_stage6 = data_stage5.map(lambda ls:[word.strip('''"''') for word in ls])
    # print(data_stage6.take(3))

    cate_dict = {}
    cate_index_dict = {}
    for features in data_stage6.collect():
        if features[0] not in cate_dict:
            cate_dict[features[0]] = 1
            cate_index_dict[features[0]] = len(cate_index_dict)
        else:
            continue
    cate_number = len(cate_dict)
    # print(cate_number,len(cate_index_dict))

    OneHot_dict = {}
    for features in data_stage6.collect():
        ori_vector = [float(0) for i in range(cate_number)]
        ori_vector[cate_index_dict[features[0]]] = float(1)
        feature_vector = ori_vector
        OneHot_dict[features[0]] = feature_vector
    # print(OneHot_dict)
    data_stage7 = data_stage6.map(lambda ls:ancillary_func(ls=ls,dict=OneHot_dict))
    return data_stage7

def ancillary_func2(ls):
    non_cate_ls = []
    for word in ls[1:len(ls)-1]:
        if word == '?' or float(word) < 0:
            non_cate_ls.append(float(0))
        else:
            non_cate_ls.append(float(word))
    flatterned = np.hstack((ls[0],non_cate_ls))
    return flatterned

def data_labeling(rdd):
    data_labeled = rdd.map(lambda ls:LabeledPoint(
        label=float(ls[-1]),
        features=ancillary_func2(ls=ls)
    ))
    return data_labeled

def data_split(rdd):
    (trainData,validationData,testData) = rdd.randomSplit([8,1,1])
    print(trainData.count(),validationData.count(),testData.count())
    return trainData,validationData,testData

def data_standardlize(rdd):
    label_data = rdd.map(lambda p:p.label)
    feature_data = rdd.map(lambda p:p.features)
    stdScaler = StandardScaler(withMean=False,withStd=True).fit(feature_data)
    Scaler_feature_RDD = stdScaler.transform(feature_data)
    labelpoint_data = label_data.zip(Scaler_feature_RDD)
    labelpoint_RDD = labelpoint_data.map(lambda pair:LabeledPoint(label=pair[0],features=pair[1:]))
    return labelpoint_RDD

def modeling_pred_accu(train_data,test_data):
    model_NBM = NaiveBayes.train(data=train_data)
    testData_features = test_data.map(lambda p:p.features)
    pred_result = model_NBM.predict(testData_features)
    flattens = pred_result.zip(test_data.map(lambda p: p.label))
    metrics = BinaryClassificationMetrics(flattens)
    print('AUC=', metrics.areaUnderROC)

def data_examine(rdd):
    examine_dict = {}
    for item in rdd.collect():
        if len(item) not in examine_dict:
            examine_dict[len(item)] = 1
        else:
            examine_dict[len(item)] += 1
    return len(examine_dict)

if __name__ == '__main__':
    print('Note that while applying the Naive-Bayesian-Model, it is important that the features should be all positive!!!')
    conf = SparkConf().setAppName('Spark_SVM')
    sc = SparkContext(conf=conf)
    data_propossessed = data_propossessing(sc=sc)
    print(data_examine(data_propossessed))
    print('data_propossessed:',data_propossessed.take(5))
    data_labeled = data_labeling(data_propossessed)
    # dc = {}
    # for item in data_labeled.collect():
    #     if len(item.features) not in dc:
    #         dc[len(item.features)] = 1
    #     else:
    #         dc[len(item.features)] += 1
    # print(dc)
    print('data_labeled:',data_labeled.take(3))
    data_labeled_std = data_standardlize(data_labeled)
    print('data_labeled_std:',data_labeled_std.take(3))
    trainData,validationData,teatData = data_split(data_labeled_std)
    modeling_pred_accu(train_data=trainData,test_data=teatData)
