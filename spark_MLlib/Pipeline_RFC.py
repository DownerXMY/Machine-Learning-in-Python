from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf,col

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier

import matplotlib.image as imgplt
from matplotlib import pyplot as plt
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['lines.linewidth'] = 3

def data_preparing(spark):
    stumble_data = spark.read.format('csv') \
        .option('header','true') \
        .option('delimiter','\t') \
        .load('/Users/mingyuexu/PycharmProjects/demo/learning/spark_demo_files/sources/train.tsv')
    # print(stumble_data.show())
    print('Similarly, we need to transfer the data into proper form:')
    # print(stumble_data.schema[0].name)
    raw_header = [stumble_data.schema[item].name for item in range(len(stumble_data.schema))]
    features_and_label = raw_header[3:len(raw_header)]
    # print(features_and_label)
    information_data = stumble_data.select('alchemy_category', 'alchemy_category_score', 'avglinksize',
                                           'commonlinkratio_1', 'commonlinkratio_2', 'commonlinkratio_3',
                                           'commonlinkratio_4', 'compression_ratio', 'embed_ratio', 'framebased',
                                           'frameTagRatio', 'hasDomainLink', 'html_ratio', 'image_ratio', 'is_news',
                                           'lengthyLinkDomain', 'linkwordscore', 'news_front_page',
                                           'non_markup_alphanum_characters', 'numberOfLinks', 'numwords_in_url',
                                           'parametrizedLinkRatio', 'spelling_errors_ratio', 'label')
    information_data.createOrReplaceTempView('stumble')
    SQL_information_table = spark.sql(('SELECT * FROM stumble LIMIT 10'))
    SQL_information_table.show()
    return information_data

def ancilarry_func(x):
    return (0 if x == '?' else x)
corrected_func = udf(ancilarry_func)

def data_strengthening(DF):
    data_corrected = DF.select(['alchemy_category']+[corrected_func(col(column)).cast('double').alias(column) for column in DF.columns[2:]])
    return data_corrected

def pipeline_dealing(stage1,stage2,stage3,stage4,data_corrected,evaluator):
    trainData,testData = data_corrected.randomSplit([0.8,0.2])
    pipeline = Pipeline(stages=[stage1,stage2,stage3,stage4])
    pipeline_model = pipeline.fit(trainData)
    model_stages_show = pipeline_model.stages[3].toDebugString
    pred_result = pipeline_model.transform(testData)
    auc = evaluator.evaluate(pred_result)
    return model_stages_show,pred_result,auc

if __name__ == '__main__':
    sql_spark = SparkSession \
    .builder \
    .appName('Pipeline_RFC') \
    .getOrCreate()

    print('Random Forest Classification:')
    fig = plt.figure()
    img = imgplt.imread('/Users/mingyuexu/PycharmProjects/demo/RFC_introduction.png')
    plt.imshow(img)
    plt.title('RFC Introduction')
    plt.show()

    information_data = data_preparing(spark=sql_spark)
    data_corrected = data_strengthening(DF=information_data)
    print('preparing for the pipeline process:')
    stringindexer = StringIndexer(
        inputCol='alchemy_category',
        outputCol='category_number'
    )
    onehotencoder = OneHotEncoder(
        dropLast=False,
        inputCol='category_number',
        outputCol='category_OneHotEncoding'
    )
    vectorassembler = VectorAssembler(
        inputCols=['category_OneHotEncoding']+[data_corrected.schema[item].name for item in range(len(data_corrected.schema))[1:len(data_corrected.schema)-1]],
        outputCol='features'
    )
    classifier = RandomForestClassifier(
        featuresCol='features',
        labelCol='label',
        impurity='gini',
        numTrees=10,
        maxDepth=10,
        maxBins=14
    )
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol='rawPrediction',
        labelCol='label',
        metricName='areaUnderROC'
    )
    model_stages_show,pred_result,auc = pipeline_dealing(stage1=stringindexer,
                     stage2=onehotencoder,
                     stage3=vectorassembler,
                     stage4=classifier,
                     data_corrected=data_corrected,
                     evaluator=evaluator)
    print('The procedures of the RFC-model:',model_stages_show)
    print(pred_result.select(['probability','prediction']).take(10))
    print('area under ROC:',auc)