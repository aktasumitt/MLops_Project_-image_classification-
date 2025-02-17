from src.components.data_transformation.create_dataset import Create_Dataset
from torchvision.transforms import transforms
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.artifact_entity import DataTransformationArtifact
from src.entity.config_entity import DataTransformationConfig
from src.utils import load_json,save_obj

from src.logging.logger import logger
from src.exception.exception import ExceptionBlock,sys


class DataTransformation():
    def __init__(self,config:DataTransformationConfig,data_ingestion_artifact:DataIngestionArtifact):
        self.config=config
        self.data_ingestion_artifact=data_ingestion_artifact
    
    def transformer(self):
        try:
            
            train_transform=transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize((self.config.resize_size,self.config.resize_size)),
                                        transforms.Normalize((0.5,),(0.5,)),
                                        transforms.RandomVerticalFlip(p=0.5),
                                        transforms.RandomRotation(30),
                                        ])
            
            valid_transform=transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((self.config.resize_size,self.config.resize_size)),
                                            transforms.Normalize((0.5,),(0.5,))])
            
            test_transform=transforms.Compose([transforms.ToTensor(),
                                            transforms.Resize((self.config.resize_size,self.config.resize_size)),
                                            transforms.Normalize((0.5,),(0.5,))])
            
            return train_transform,valid_transform,test_transform
        
        except Exception as e:
            raise ExceptionBlock(e,sys)
        
    def initiate_transforms(self):
        try: 
            
            train_dataset_path_dict=load_json(self.data_ingestion_artifact.train_data_dict_path)
            test_dataset_path_dict=load_json(self.data_ingestion_artifact.test_data_dict_path)
            valid_dataset_path_dict=load_json(self.data_ingestion_artifact.valid_data_dict_path)
            
            train_transform,valid_transform,test_transform=self.transformer()
            
            train_dataset=Create_Dataset(dataset_path_dict=train_dataset_path_dict,
                                        transformer=train_transform)
            
            test_dataset=Create_Dataset(dataset_path_dict=test_dataset_path_dict,
                                        transformer=valid_transform)
            
            valid_dataset=Create_Dataset(dataset_path_dict=valid_dataset_path_dict,
                                        transformer=test_transform)
            
            save_obj(train_dataset,self.config.save_path_train_dataset)
            logger.info(f"Train dataset was saved on [ {self.config.save_path_train_dataset} ]")
            
            save_obj(valid_dataset,self.config.save_path_valid_dataset)
            logger.info(f"Validation dataset was saved on [ {self.config.save_path_train_dataset} ]")
            
            save_obj(test_dataset,self.config.save_path_test_dataset)
            logger.info(f"Test dataset was saved on [ {self.config.save_path_train_dataset} ]")
            
            
            data_transformation_artifact=DataTransformationArtifact(train_dataset_paths=self.config.save_path_train_dataset,
                                                                    test_dataset_paths=self.config.save_path_test_dataset,
                                                                    validation_dataset_path=self.config.save_path_valid_dataset)
            return data_transformation_artifact
        
        except Exception as e:
            raise ExceptionBlock(e,sys)
        