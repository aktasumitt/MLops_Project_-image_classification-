from src.components.Prediction.prediction import Prediction
from src.config.configuration import Configuration
from src.logging.logger import logger
from src.exception.exception import ExceptionBlock, sys


class PredictionPipeline():
    def __init__(self):
        self.configuration = Configuration()

    def prediction(self, prediction_config):
        try:
            prediction = Prediction(prediction_config)
            results = prediction.predict_and_save_result()
            return results

        except Exception as e:
            raise ExceptionBlock(e, sys)

    def run_prediction_pipeline(self):
        try:
            logger.info("***** Prediction is starting... *****")
            prediction_config = self.configuration.get_prediction_configs()
            results = self.prediction(prediction_config)
            logger.info("***** Prediction finished... *******")
            return results

        except Exception as e:
            raise ExceptionBlock(e, sys)


if __name__ == "__main__":

    prediction_pipeline = PredictionPipeline()
    prediction_pipeline.run_prediction_pipeline()
