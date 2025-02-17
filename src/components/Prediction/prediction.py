from torch.utils.data import DataLoader
from src.entity.config_entity import PredictionConfig
from src.components.Prediction.model_prediction import predict
from src.components.Prediction.preprocess_data import PreprocessPredictionData
from src.utils import save_as_json,load_obj,load_checkpoints,load_json

from src.logging.logger import logger
from src.exception.exception import ExceptionBlock,sys


class Prediction():
    
    def __init__(self,config:PredictionConfig):
        self.config=config
        self.model=load_obj(path=config.final_model_path).to(config.device)

    def load_dataset(self):
        try:
            preprocess_data=PreprocessPredictionData(resize_size=self.config.image_size)
            img_path_dict=preprocess_data.get_labels_and_image_paths(local_data_folder=self.config.local_data_folder)
                    
            prediction_dataset=preprocess_data.data_transformation(images_path_dict=img_path_dict)
            prediction_dataloader=DataLoader(dataset=prediction_dataset,batch_size=self.config.batch_size,shuffle=False)
            
            return prediction_dataloader,img_path_dict
        except Exception as e:
           raise ExceptionBlock(e,sys)
    
    def initiate_prediction(self):
        try:
            predict_dataloader,img_path_dict=self.load_dataset()            
            
            prediction_labels=predict(prediction_dataloader=predict_dataloader,
                                        Model=self.model,
                                        batch_size=self.config.batch_size,
                                        devices=self.config.device)
            
            
            return prediction_labels,img_path_dict
        
        except Exception as e:
           raise ExceptionBlock(e,sys)
    
    def predict_and_save_result(self):
        try:
            label_names=self.config.labels
            label_names={v:k for k,v in label_names.items()}
            
            prediction_labels,img_path_dict=self.initiate_prediction()

            results={}
            for i,img_path in enumerate(list(img_path_dict.keys())):
                label=prediction_labels[i]
                results[str(img_path)]=label_names[(label)]
                
            save_as_json(results,save_path=self.config.save_path_result_path)
            logger.info(f"Prediction results is saved on [ {self.config.save_path_result_path} ]")
            
            return list(results.values())    
             
        except Exception as e:
           raise ExceptionBlock(e,sys)    
        
    
    
        