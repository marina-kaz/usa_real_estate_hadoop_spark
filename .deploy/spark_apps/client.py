import logging
from os.path import join
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_unixtime, to_timestamp
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.ml import Pipeline
from pyspark.ml.regression import FMRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder, StandardScaler
from tap import Tap

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


def get_data():
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




if __name__ == "__main__":
    args = CLI(underscores_to_dashes=True).parse_args()
    logger = get_logger(args.log_prefix)

    logger.info('Starting Application...')
    spark = (SparkSession.builder
            .appName("SparkFMApp")
            .getOrCreate())
    logger.info('Application started!')


    data = get_data()
    logger.info('Read data!')

    data = process_data(data)
    logger.info('Processed data!')

    logger.info('Finishing!')
    spark.stop()
