from src.entity.config_entity import TestingConfig
from src.components.testing.test_model import model_testing
from src.utils import save_as_json,load_obj,load_checkpoints,save_obj

from torch.utils.data import DataLoader
import torch

from src.logging.logger import logger
from src.exception.exception import ExceptionBlock,sys

class Testing():
    def __init__(self,config:TestingConfig):
        self.config=config
        
        self.model=load_obj(self.config.final_model_path).to(self.config.device)
        

    def load_object(self):
        try:
            test_dataset=load_obj(self.config.test_dataset_path)
            test_dataloader=DataLoader(test_dataset,batch_size=self.config.batch_size,shuffle=True)

            loss_fn=torch.nn.CrossEntropyLoss()        
            
            return test_dataloader,loss_fn
        
        except Exception as e:
            raise ExceptionBlock(e,sys)
            
    def initiate_testing(self):
        try:
            
            test_dataloader,loss_fn = self.load_object()
            
            # Load Checkpoints if u want
            if self.config.load_checkpoints==True:
                load_checkpoints(path=self.config.load_checkpoints,model=self.model)    
            
            test_loss, test_acc = model_testing(test_dataloader=test_dataloader,
                                                loss_fn=loss_fn,
                                                Model=self.model,
                                                devices=self.config.device)
            
            metrics={"Test_loss":test_loss, "Test_acc":test_acc}

            # Save tested model if you want (you can use this after load spesific checkpoints)
            if self.config.save_tested_model==True:
                save_obj(self.model,save_path=self.config.tested_model_save_path)              
  
            save_as_json(data=metrics,save_path=self.config.save_result_path)
            logger.info(f"Testing model is completed. Test results was saved on [ {self.config.save_result_path} ]")
            
            return metrics
        
        except Exception as e:
            raise ExceptionBlock(e,sys)