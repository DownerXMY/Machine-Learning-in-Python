from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.sql.functions import col

# The following are Pipeline modules
# One may find its amazement by noticing that all the need stuffs are included!!!
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# we here directly apply the DF tool to read the data, which is much more efficient.
def data_preparing(spark):
    stumble_data = spark.read.format('csv') \
    .option('header','true') \
    .option('delimiter','\t') \
    .load('/Users/mingyuexu/PycharmProjects/demo/learning/spark_demo_files/sources/train.tsv')
    print(stumble_data.show())
    print('Similarly, we need to transfer the data into proper form:')
    # print(stumble_data.schema[0].name)
    raw_header = [stumble_data.schema[item].name for item in range(len(stumble_data.schema))]
    features_and_label = raw_header[3:len(raw_header)]
    # print(features_and_label)
    information_data = stumble_data.select('alchemy_category', 'alchemy_category_score', 'avglinksize', 'commonlinkratio_1', 'commonlinkratio_2', 'commonlinkratio_3', 'commonlinkratio_4', 'compression_ratio', 'embed_ratio', 'framebased', 'frameTagRatio', 'hasDomainLink', 'html_ratio', 'image_ratio', 'is_news', 'lengthyLinkDomain', 'linkwordscore', 'news_front_page', 'non_markup_alphanum_characters', 'numberOfLinks', 'numwords_in_url', 'parametrizedLinkRatio', 'spelling_errors_ratio', 'label')
    information_data.createOrReplaceTempView('stumble')
    SQL_information_table = spark.sql(('SELECT * FROM stumble LIMIT 10'))
    SQL_information_table.show()
    return information_data

# Moreover, note that there are many '?' in the dataset!
# And also we need to transfer the category into OneHotencoding.
# To realize the above two requirements, we need new tools!
# We called the 'self-defined function'.

def transfer_func(x):
    return ('0' if x == '?' else x)
transfer_function = udf(transfer_func)

def data_properize(DF):
    print('transfer to Double:')
    data_stage1 = DF.select(['alchemy_category']+[transfer_function(col(column)).cast('double').alias(column) for column in DF.columns[2:]])
    print(data_stage1.show(10))
    return data_stage1

def data_spliting(DF):
    trainData,testData = DF.randomSplit([0.8,0.2])
    trainData.cache()
    testData.cache()
    return trainData,testData

def data_fitting(transfer1,transfer2,transfer3,classifier,trainData,testData,wholeData):
    print('after category transferred:')
    categoty_Transfomer = transfer1.fit(wholeData)
    data_trans_whole = categoty_Transfomer.transform(wholeData)
    data_trans_train = categoty_Transfomer.transform(trainData)
    data_trans_test = categoty_Transfomer.transform(testData)
    data_trans_train.show(5)

    print('after category OneHotencoded:')
    OneHot_transfer = transfer2.fit(data_trans_whole)
    data_trans_whole2 = OneHot_transfer.transform(data_trans_whole)
    data_trans_train2 = OneHot_transfer.transform(data_trans_train)
    data_trans_test2 = OneHot_transfer.transform(data_trans_test)
    data_trans_train2.show(5)

    print('after features assembled:')
    data_trans_whole3 = transfer3.transform(data_trans_whole2)
    data_trans_train3 = transfer3.transform(data_trans_train2)
    data_trans_test3 = transfer3.transform(data_trans_test2)
    data_trans_train3.show(5)

    tree_model = classifier.fit(data_trans_whole3)
    data_trans_whole4 = tree_model.transform(data_trans_whole3)
    return tree_model

def pipeline_process(stringindexer,onehotencoder,vectorassembler,classifier,trainData,testData):
    pipeline = Pipeline(stages=[stringindexer,
                                onehotencoder,
                                vectorassembler,
                                classifier])
    print(pipeline.getStages())
    print('model fitting:')
    pipeline_Model = pipeline.fit(trainData)
    # Notice that the model will form the Tree model in stage3, hence we can check that stage:
    print(pipeline_Model.stages[3])
    # 还可以看看决策树每一个阶段的判定标准和流程：
    print(pipeline_Model.stages[3].toDebugString)
    print('model prediction:')
    pred_result = pipeline_Model.transform(testData)
    return pred_result

def prediction_accuracy(evaluator,pred_result):
    auc = evaluator.evaluate(pred_result)
    return auc

if __name__ == '__main__':
    sql_spark = SparkSession \
    .builder \
    .appName('ML_in_DF') \
    .getOrCreate()
    print('From now on, we introduce an important gear called "Pipeline"')
    raw_information_data = data_preparing(spark=sql_spark)
    data_proper = data_properize(DF=raw_information_data)
    trainData,testData = data_spliting(data_proper)
    categotyIndexer = StringIndexer(
        inputCol='alchemy_category',
        outputCol='category_transformation'
    )
    OneHot_encoder = OneHotEncoder(
        dropLast=False,
        inputCol='category_transformation',
        outputCol='category_OneHotEncoding'
    )
    Vector_assembler = VectorAssembler(
        inputCols=['category_OneHotEncoding']+[data_proper.schema[item].name for item in range(len(data_proper.schema))][1:-1],
        outputCol='features'
    )
    classifier = DecisionTreeClassifier(
        labelCol='label',
        featuresCol='features',
        impurity='gini',
        maxDepth=10,
        maxBins=14
    )

    # model1 = data_fitting(transfer1=categotyIndexer,
    #                  transfer2=OneHot_encoder,
    #                  transfer3=Vector_assembler,
    #                  classifier=classifier,
    #                  trainData=trainData,
    #                  testData=testData,
    #                  wholeData=data_proper)

    print('Actually, the "data_fitting" shows the specific procedures of the Pipeline')
    print('However, it is quite simple for us to establish the Pipeline process')
    # we do not need to have so many data stages, but only th merge them together:
    pred_result = pipeline_process(stringindexer=categotyIndexer,
                     onehotencoder=OneHot_encoder,
                     vectorassembler=Vector_assembler,
                     classifier=classifier,
                     trainData=trainData,
                     testData=testData)
    print(pred_result.select('probability','prediction').take(10))
    # probability is of the form DenseVector([P{predicted to be 0},P{predicted to be 1}])
    evaluator = BinaryClassificationEvaluator(
        rawPredictionCol='rawPrediction',
        labelCol='label',
        metricName='areaUnderROC'
    )
    auc = prediction_accuracy(evaluator=evaluator,pred_result=pred_result)
    print('area under ROC:',auc)
