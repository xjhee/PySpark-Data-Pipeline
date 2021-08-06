import pyspark 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np 




def split_data(spark):
    df_test = spark.read.csv(r"query.csv", header=True)
    df_train = spark.read.format('mongo')\
        .option('spark.mongodb.input.uri', 'mongodb://127.0.0.1:27017/Quake.quakes').load()
    df_test_clean = df_test['time', 'latitude', 'longitude', 'mag', 'depth']
    df_test_clean = df_test_clean.withColumnRenamed('time', 'Date')\
        .withColumnRenamed('latitude', 'Latitude')\
        .withColumnRenamed('longitude', 'Longitude')\
        .withColumnRenamed('mag', 'Magnitude')\
        .withColumnRenamed('depth', 'Depth')

    df_test_clean = df_test_clean.withColumn('Latitude', df_test_clean['Latitude'].cast(DoubleType()))\
        .withColumn('Longitude', df_test_clean['Longitude'].cast(DoubleType()))\
        .withColumn('Depth', df_test_clean['Depth'].cast(DoubleType()))\
        .withColumn('Magnitude', df_test_clean['Magnitude'].cast(DoubleType()))

    df_testing = df_test_clean['Latitude', 'Longitude', 'Magnitude', 'Depth']
    df_training = df_train['Latitude', 'Longitude', 'Magnitude', 'Depth']

    df_training = df_training.dropna()
    df_testing = df_testing.dropna()
    return df_training, df_testing



def random_forest(df_training, df_testing):
    assembler = VectorAssembler(inputCols=['Latitude', 'Longitude', 'Depth'], outputCol='features')
    model_reg = RandomForestRegressor(featuresCol='features', labelCol='Magnitude')
    pipeline = Pipeline(stages=[assembler, model_reg])

    model = pipeline.fit(df_training)
    pred_results = model.transform(df_testing)
    evaluator = RegressionEvaluator(labelCol='Magnitude', predictionCol='prediction', metricName='rmse')
    rmse = evaluator.evaluate(pred_results)


    df_pred_results = pred_results['Latitude', 'Longitude', 'prediction']
    df_pred_results = df_pred_results.withColumnRenamed('prediction', 'Pred_Magnitude')
    df_pred_results = df_pred_results.withColumn('Year', lit(2017))\
        .withColumn('RMSE', lit(rmse))

    df_pred_results.write.format('mongo')\
        .mode('overwrite')\
        .option('spark.mongodb.output.uri', 'mongodb://127.0.0.1:27017/Quake.pred_results').save()

    print(df_pred_results.show(5))

    print('INFO: Job ran successfully')
    print('')



def run_ml_pipeline():
    spark = SparkSession\
    .builder\
    .master('local[2]')\
    .appName('quakes_ml')\
    .config('spark.jars.packages", "org.mongodb.spark:mongo-spark-connector_2.12:2.4.1')\
    .getOrCreate()
    df_training, df_testing = split_data(spark)
    random_forest(df_training, df_testing)



if __name__ == '__main__':
    run_ml_pipeline()