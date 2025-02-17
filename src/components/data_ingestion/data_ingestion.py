from src.utils import save_as_json,save_as_yaml
import random
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from pathlib import Path
from src.logging.logger import logger
from src.exception.exception import ExceptionBlock
import sys


class DataIngestion():
    
    def __init__(self,config:DataIngestionConfig):
        self.config= config
        self.labels= self.config.labels
        self.dataset_folder_path=Path(self.config.local_data_folder)
        
    def get_train_images_data_path(self):
        try:
            
            train_images_path_dict={}
            
            for data_type in self.dataset_folder_path.glob("*"):
                
                if data_type.name == "train":
                    for label_path in Path(data_type).glob("*"):
                        label=label_path.name
                        label_idx=self.labels[label]

                        for img_path in Path(label_path).glob("*"):
                            img_path=str(img_path)
                            train_images_path_dict[img_path]=label_idx    

            return train_images_path_dict

        except Exception as e:
            raise ExceptionBlock(e,sys)
    
    
    def get_test_images_data_path(self):
        try:
            
            test_images_path_dict={}
            
            for data_type in self.dataset_folder_path.glob("*"):
                
                if  data_type.name == "test":
                    for label_path in Path(data_type).glob("*"):
                        label=label_path.name
                        label_idx=self.labels[label]

                        for img_path in Path(label_path).glob("*"):
                            img_path=str(img_path)
                            test_images_path_dict[img_path]=label_idx
                    
                    save_as_json(test_images_path_dict,self.config.save_test_img_path)   
                    logger.info(f"Test images path dict was saved on [ {self.config.save_test_img_path} ]")

        except Exception as e:
            raise ExceptionBlock(e,sys)
        
    def get_valid_data_from_train(self,data_dict:dict):
        
        try:
            image_list=list(data_dict.items())
            
            random.seed(a=self.config.random_state)
            random.shuffle(image_list)
            
            split_len=int(len(image_list)*self.config.valid_split_rate)
            valid_img_paths=dict(image_list[:split_len])
            train_img_paths=dict(image_list[split_len:])
            
            save_as_json(valid_img_paths,self.config.save_valid_img_path)
            logger.info(f"Valid images path dict was saved on [ {self.config.save_valid_img_path} ]")

            save_as_json(train_img_paths,self.config.save_train_img_path)
            logger.info(f"Train images path dict was saved on [ {self.config.save_train_img_path} ]")
       
        except Exception as e:
            raise ExceptionBlock(e,sys)    
    
    def initiate_data_ingestion(self):
        try:
            
            train_img_dict=self.get_train_images_data_path()
            self.get_valid_data_from_train(train_img_dict)
            self.get_test_images_data_path()
            
            data_ingestion_artifacts=DataIngestionArtifact(train_data_dict_path=self.config.save_train_img_path,
                                                            valid_data_dict_path=self.config.save_valid_img_path,
                                                            test_data_dict_path=self.config.save_test_img_path,)
            return data_ingestion_artifacts
        
        except Exception as e:
            raise ExceptionBlock(e,sys)


