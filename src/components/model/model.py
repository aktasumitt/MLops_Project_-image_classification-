import torch.nn as nn
from src.entity.config_entity import ModelConfig
from src.entity.artifact_entity import ModelArtifact
from src.utils import save_obj
from src.logging.logger import logger
from src.exception.exception import ExceptionBlock,sys

class CNNModel(nn.Module):
    def __init__(self,channel_size,img_size,label_size):
        super().__init__()
        self.channel_size=channel_size
        
        self.conv_seq=nn.Sequential(nn.Conv2d(in_channels=channel_size,out_channels=32,kernel_size=3,padding=1,stride=1,bias=False),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2),
                                    
                                    nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1,stride=1,bias=False),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2),
                                    
                                    nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1,stride=1,bias=False),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2,stride=2),
                                    
                                    nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,padding=1,stride=1,bias=False),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    )
        
        self.in_feature=int(((img_size//8)**2)*256)
        
        self.linear_seq=nn.Sequential(nn.Linear(in_features=self.in_feature,out_features=1024),
                                      nn.Dropout(0.1),
                                      nn.ReLU(),
                                      nn.Linear(in_features=1024,out_features=512),
                                      nn.Dropout(0.1),
                                      nn.ReLU(),
                                      nn.Linear(in_features=512,out_features=128),
                                      nn.Dropout(0.1),
                                      nn.ReLU(),
                                      nn.Linear(in_features=128,out_features=32),
                                      nn.ReLU(),
                                      nn.Linear(32,label_size)
                                      )
        
        
    def forward(self,data):
        try:
            x1=self.conv_seq(data)
                    
            x2=x1.view(-1,self.in_feature)
            
            x3=self.linear_seq(x2)
            
            return x3
        
        except Exception as e:
           raise ExceptionBlock(e,sys)
    
    

class ModelIngestion():
    def __init__(self,config:ModelConfig):
        
        self.config=config
        
    def create_model(self):
        try: 
            model=CNNModel(channel_size=self.config.channel_size,
                        img_size=self.config.img_size,
                        label_size=self.config.label_size)
            
            save_obj(model,save_path=self.config.save_path_model)
            logger.info(f"Model object were saved on [ {self.config.save_path_model} ]")
            
            
            model_artifact=ModelArtifact(model_path=self.config.save_path_model)
            return model_artifact 
              
        except Exception as e:
            raise ExceptionBlock(e,sys)
            