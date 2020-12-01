from pyspark.conf import SparkConf
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree
from pyspark.mllib.evaluation import BinaryClassificationMetrics

import pandas as pd
import numpy as np

print('Note that the .tsv file can be read by pandas!')
train_data = pd.read_csv('/Users/mingyuexu/PycharmProjects/demo/learning/spark_demo_files/sources/train.tsv',sep='\t')
test_data = pd.read_csv('/Users/mingyuexu/PycharmProjects/demo/learning/spark_demo_files/sources/test.tsv',sep='\t')
print(train_data.loc[0:5],test_data.loc[0:5])
print('However, here we can also read the .tsv by sc.textFile!')

def RDD_to_DF(rdd):
    spark = SparkSession \
            .builder \
            .appName('spark_Dtree') \
            .getOrCreate()
    rdd1 = rdd.filter(lambda tp:tp[1] != 'u')
    rdd2 = rdd.filter(lambda tp:tp[1] == 'u')

    RDD_index_list = rdd2.map(lambda tp: tp.split('\t')).collect()
    RDD_String = ''
    for lst in RDD_index_list:
        for index in lst:
            index = index.strip('''"''')
            RDD_String = RDD_String + index + ' '
    # print(RDD_String)

    table = rdd1.map(lambda tp:tp.split('\t')).collect()
    pre_RDD = [group for group in table]
    get_RDD = sc.parallelize(pre_RDD)
    table_cor = get_RDD.map(lambda ls:tuple([word.strip('''"''') for word in ls]))
    fields = [StructField(field_name, StringType(), True) for field_name in RDD_String.split(' ')[0:27]]
    schema = StructType(fields)
    get_DF = spark.createDataFrame(table_cor, schema)
    get_DF.createOrReplaceTempView('stumble')
    sqlDF2 = spark.sql(('SELECT * FROM stumble WHERE label == 1'))
    sqlDF2.show()

def ancillary_func(ls,dt):
    new_ls = [dt[ls[0]]]
    new_ls.extend(ls[1:])
    return new_ls

def train_data_propossesssing(rdd,sc):
    header = rdd.first()
    raw_data = rdd.filter(lambda x:x != header)
    data_stage_one = raw_data.map(lambda tp:tp.split(','))
    # print(data_stage_one.collect())
    data_stage_two = data_stage_one.map(lambda tp:[word.split('\t') for word in tp][-1])
    # print(data_stage_two.collect())
    data_stage_three = data_stage_two.map(lambda ls:[word.strip('''"''') for word in ls][1:])
    # print(data_stage_three.collect())
    data_stage_three_ls = []
    for tp in data_stage_three.collect():
        ls = [tp[0]]
        for word in tp[1:]:
            if word == '?':
                ls.append(0.0)
            else:
                ls.append(float(word))
        data_stage_three_ls.append(ls)
    data_stage_three_fix = sc.parallelize(data_stage_three_ls)
    # get the categories into oneHotencodings:
    cate_list = []
    for ls in data_stage_three.collect():
        cate_list.append(ls[0])
    cate_num = len(set(cate_list))
    OH_dict = {}
    for cate in cate_list:
        OH_vector = np.zeros(cate_num)
        if cate not in OH_dict:
            OH_vector[len(OH_dict)] = 1
            OH_dict[cate] = OH_vector
        else:
            continue

    data_stage_four = data_stage_three_fix.map(lambda tp:ancillary_func(tp,OH_dict))
    print('data examples:',data_stage_four.take(1))
    return data_stage_four

def establish_form(rdd):
    labelpoint_RDD = rdd.map(lambda r: LabeledPoint(
        label=float(r[-1]),
        features=np.hstack((r[0],r[1:len(r)-1]))))
    print(labelpoint_RDD.collect())
    return labelpoint_RDD

def split_data(rdd):
    (trainData,validationData,testData) = rdd.randomSplit([8,1,1])
    print(trainData.count(),validationData.count(),testData.count())
    return trainData,validationData,testData

def model_pred_accu(trainData,testData):
    model = DecisionTree.trainClassifier(
        trainData,numClasses=2,categoricalFeaturesInfo={},
        impurity='entropy',maxDepth=5,maxBins=5
    )
    testData_features = testData.map(lambda p:p.features)
    pred_result = model.predict(testData_features)
    flattens = pred_result.zip(testData.map(lambda p:p.label))
    metrics = BinaryClassificationMetrics(flattens)
    print('AUC=',metrics.areaUnderROC)

if __name__ == '__main__':
    conf = SparkConf().setAppName('spark_Dtree')
    sc = SparkContext(conf=conf)
    raw_train_data = sc.textFile('/Users/mingyuexu/PycharmProjects/demo/learning/spark_demo_files/sources/train.tsv')
    raw_test_data = sc.textFile('/Users/mingyuexu/PycharmProjects/demo/learning/spark_demo_files/sources/test.tsv')
    # RDD_to_DF(raw_train_data)
    # RDD_to_DF(raw_test_data)
    print('-------------------------------')
    print('spark_Dtree:')
    # print(raw_train_data.take(3))
    raw_RDD = train_data_propossesssing(raw_train_data,sc=sc)
    labeled_RDD = establish_form(raw_RDD)
    trainData,validationData,testData = split_data(labeled_RDD)
    # Take into memory to accelerate the codes.
    trainData.persist()
    validationData.persist()
    testData.persist()

    model_pred_accu(trainData,testData)
