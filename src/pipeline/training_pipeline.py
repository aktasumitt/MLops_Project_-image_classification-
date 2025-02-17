from src.components.data_ingestion.data_ingestion import DataIngestion
from src.components.data_transformation.transforms import DataTransformation
from src.components.model.model import ModelIngestion
from src.components.training.training import Training
from src.config.configuration import Configuration

from src.logging.logger import logger
from src.exception.exception import ExceptionBlock,sys

class TrainingPipeline():
    def __init__(self):
        self.configuration=Configuration()
        
    def data_ingestion(self,data_ingestion_config):
        try:
            logger.info("********  Data Ingestion is starting... ********")
            data_ingestion=DataIngestion(data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        
        except Exception as e:
            ExceptionBlock(e,sys)
    
    def data_transformation(self,data_transformation_config,data_ingestion_artifact):
        try:
            logger.info("********  Data Transformation is starting...  *********")
            data_transformation=DataTransformation(data_transformation_config,data_ingestion_artifact)
            data_transformation_artifact=data_transformation.initiate_transforms()
            return data_transformation_artifact
        
        except Exception as e:
            ExceptionBlock(e,sys)
            
    def model_initiate(self,model_config):
        try:
            logger.info("********  Model Initialization is starting...  ********")
            model_ingestion=ModelIngestion(model_config)
            model_artifact=model_ingestion.create_model()
            return model_artifact
        
        except Exception as e:
            ExceptionBlock(e,sys)
        
    def training(self,training_config,data_transformation_artifact,model_artifact):
        try:
            logger.info("********  Model Training is starting...  ********")
            training=Training(training_config,data_transformation_artifact,model_artifact)
            training_artifact=training.start_training_with_mlflow()
            return training_artifact
        
        except Exception as e:
            ExceptionBlock(e,sys)
            
    def run_training_pipeline(self):
        try:
            logger.info("********  Training pipeline is starting...  ********")
            data_ingestion_artifact = self.data_ingestion(self.configuration.get_data_ingestion_configs())
            data_transformation_artifact = self.data_transformation(data_transformation_config=self.configuration.get_data_transformation_configs(),
                                                                                data_ingestion_artifact=data_ingestion_artifact)

            model_artifacts = self.model_initiate(model_config=self.configuration.get_model_configs())

            
            training_artifacts = self.training(training_config=self.configuration.get_training_configs(),
                                                        data_transformation_artifact=data_transformation_artifact,
                                                        model_artifact=model_artifacts)

            logger.info("********  Training pipeline finished successfully...  ********")
            
            return training_artifacts
        
        except Exception as e:
            ExceptionBlock(e,sys)     
                    


    

                        
    
