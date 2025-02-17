from src.components.data_transformation.transforms import Create_Dataset
from src.components.data_ingestion.data_ingestion import DataIngestion
from torchvision.transforms import transforms
from pathlib import Path
from src.exception.exception import ExceptionBlock,sys

class PreprocessPredictionData():
    def __init__(self,resize_size):
        
        self.prediction_transformer=transforms.Compose([transforms.ToTensor(),
                                                    transforms.Resize((resize_size,resize_size)),
                                                    transforms.Normalize((0.5,),(0.5,))])
    
    
    def get_labels_and_image_paths(self,local_data_folder):
        try:
            image_path_dict={}
            
            dataset_folder_path=Path(local_data_folder)
            
            for idx,img_path in enumerate(dataset_folder_path.glob("*")):
                image_path_dict[img_path]=idx
            
            return image_path_dict
        
        except Exception as e:
            raise ExceptionBlock(e,sys)
    
    def data_transformation(self,images_path_dict):
        try: 
            transformed_data=Create_Dataset(images_path_dict,transformer=self.prediction_transformer)
            return transformed_data

        except Exception as e:
            raise ExceptionBlock(e,sys)
    
    
    