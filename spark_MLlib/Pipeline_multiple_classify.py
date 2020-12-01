from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import col

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import DecisionTreeClassifier

def data_preparing(sc):
    raw_data = sc.textFile('/usr/local/Cellar/apache-spark/3.0.0/libexec/sources/covtype.data')
    data_stage1 = raw_data.map(lambda x:x.split(','))
    # print(data_stage1.take(10))
    fieldnum = len(data_stage1.first())
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
    get_DF = spark.createDataFrame(data_stage1,schema)
    get_DF.show()
    return get_DF

def data_to_double(DF):
    data_stage2 = DF.select([col(column).cast('double').alias(column) for column in DF.columns])
    return data_stage2

def data_corrected(DF):
    data_stage3 = DF.withColumn('Label',DF['label']-1)
    data_stage3.show(10)
    return data_stage3

def modeling_and_prediction(DF,stage1,stage2,evaluator):
    trainData,testData = DF.randomSplit([0.8,0.2])
    pipeline = Pipeline(stages=[stage1,stage2])
    pipeline_model = pipeline.fit(trainData)
    pipeline_stage_show = pipeline_model.stages[1].toDebugString
    pred_result = pipeline_model.transform(testData)
    accuracy = evaluator.evaluate(pred_result)
    return pipeline_stage_show,pred_result,accuracy

if __name__ == '__main__':
    spark = SparkSession \
    .builder \
    .appName('Multiple_classification_with_DF') \
    .getOrCreate()
    sc = spark.sparkContext

    dataDF = data_preparing(sc=sc)
    DF_doubled = data_to_double(DF=dataDF)
    DF_corrected = data_corrected(DF=DF_doubled)
    vectorassembler = VectorAssembler(
        inputCols=[DF_corrected.schema[item].name for item in range(len(DF_corrected.schema))[:-1]],
        outputCol='features'
    )
    classifier = DecisionTreeClassifier(
        featuresCol='features',
        labelCol='Label',
        maxDepth=5,
        maxBins=20
    )
    evaluator = MulticlassClassificationEvaluator(
        predictionCol='prediction',
        labelCol='Label',
        metricName='accuracy'
    )
    pipeline_model_show,pred_result,accuracy = modeling_and_prediction(DF=DF_corrected,
                            stage1=vectorassembler,
                            stage2=classifier,
                            evaluator=evaluator)
    print('Tree Model Show:',pipeline_model_show)
    print('prediction:',pred_result.select(['probability','prediction']).take(10))
    print('accuracy=',accuracy)