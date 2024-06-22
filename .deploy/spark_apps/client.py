import logging
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import FMRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from tap import Tap
# import os
from time import perf_counter
import psutil


class CLI(Tap):
    log_prefix: str = "2nodes"
    optimized: bool = False


def get_logger(prefix: str):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    console_handler.flush()

    file_handler = logging.FileHandler(f'/opt/spark/apps/{prefix}_{datetime.now()}.log')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    file_handler.flush()

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def get_data(spark):
    data_path = "/opt/spark/data/usa_real_estate.csv"
    return spark.read.csv(data_path, header=True, inferSchema=True)

def process_data(df):
    categorical_columns = ["city", "state"]
    numerical_columns = ["bed", "bath", "acre_lot", "house_size"]
    indexers = [StringIndexer(inputCol=column, outputCol=column+"_index") for column in categorical_columns]
    encoders = [OneHotEncoder(inputCol=indexer.getOutputCol(), outputCol=column+"_vec") for indexer, column in zip(indexers, categorical_columns)]
    assembler_inputs = [column+"_vec" for column in categorical_columns] + numerical_columns

    assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features_unscaled")
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features", withStd=True, withMean=True)
    pipeline = Pipeline(stages=indexers + encoders + [assembler, scaler])
    
    preprocessor = pipeline.fit(df)
    return preprocessor.transform(df)

def train_model(df):
    train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
    fm = FMRegressor(featuresCol="features", labelCol="price", maxIter=1)
    model = fm.fit(train_data)
    return model, test_data

def evaluate_model(model, test_data):
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    
    return rmse


def main(log_prefix: str, optimized: bool):
    logger = get_logger(log_prefix)
    time_start = perf_counter() 


    logger.info('Starting Application...')
    spark = (SparkSession.builder
            .appName("SparkFMApp")
            .getOrCreate())
    logger.info('Application started!')

    data = get_data(spark)
    logger.info('Read data!')

    processed_data = process_data(data)
    logger.info('Processed data!')

    logger.info('Training model...')
    model, test_data = train_model(processed_data)
    logger.info('Trained model!')

    rmse = evaluate_model(model, test_data)
    logger.info(f'Root Mean Squared Error (RMSE) on test data = {rmse}')

    time_finish = perf_counter() 
    ram = psutil.Process().memory_info().rss / (float(1024)**2)
    logger.info(f'Elapsed time: {time_finish - time_start}')
    logger.info(f'Total RAM: {ram}')

    logger.info('Finishing!')
    spark.stop()


if __name__ == "__main__":
    args = CLI(underscores_to_dashes=True).parse_args()
    main(log_prefix=args.log_prefix, optimized=args.optimized)
