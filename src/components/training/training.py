from src.components.training.model_train import model_training
from src.components.training.model_validation import model_validation
from src.entity.config_entity import TrainingConfig
from src.entity.artifact_entity import (ModelArtifact,
                                        DataTransformationArtifact,
                                        TrainingArtifact)

from src.utils import save_as_json,load_obj,load_checkpoints,save_checkpoints,save_obj
from torch.utils.data import DataLoader
import torch
from src.logging.logger import logger
from src.exception.exception import ExceptionBlock,sys
import mlflow
from mlflow.models import infer_signature
import subprocess

# import dagshub
# dagshub.init(repo_owner='umitaktas', repo_name='MLops_Project_-image_classification-', mlflow=True)


class Training():
    
    def __init__(self,config:TrainingConfig,data_transformation_artifact:DataTransformationArtifact,model_artifact:ModelArtifact):
        self.config=config
        self.data_transformation_artifact=data_transformation_artifact
        self.model_artifact=model_artifact
        
        self.model=load_obj(self.model_artifact.model_path).to(self.config.device)
        
        self.optimizer=torch.optim.Adam(self.model.parameters(),
                                        lr=self.config.lr,
                                        betas=(self.config.beta1,self.config.beta2))
        
    def load_object(self):
        try:
            train_dataset=load_obj(self.data_transformation_artifact.train_dataset_paths)
            train_dataloader=DataLoader(train_dataset,batch_size=self.config.batch_size,shuffle=True)
            
            valid_dataset=load_obj(self.data_transformation_artifact.validation_dataset_path)
            valid_dataloader=DataLoader(valid_dataset,batch_size=self.config.batch_size,shuffle=False)
            
            loss_fn=torch.nn.CrossEntropyLoss()        
            
            return train_dataloader,valid_dataloader,loss_fn
        
        except Exception as e:
            ExceptionBlock(e,sys)
    
    
    def initiate_training(self):
        try:
            result_list=[]
            
            train_dataloader,valid_dataloader,loss_fn=self.load_object()
            
            starting_epoch=1
            if self.config.load_checkpoints==True:
                starting_epoch=load_checkpoints(path=self.config.save_path_checkpoint,model=self.model,optimizer=self.optimizer)
                logger.info(f"Checkpoints were loaded. Training is starting from {starting_epoch}.epoch")
            
            # start an MLFLOW run
            for epoch in range(starting_epoch,self.config.epoch+1):
                    
                train_loss, train_acc = model_training(train_dataloader=train_dataloader,
                                                        optimizer=self.optimizer,
                                                        loss_fn=loss_fn,
                                                        Model=self.model,
                                                        device=self.config.device)
                    
                valid_loss,valid_acc = model_validation(valid_dataloader=valid_dataloader,
                                                        loss_fn=loss_fn,
                                                        Model=self.model,
                                                        device=self.config.device)
                    
                save_checkpoints(save_path=self.config.save_path_checkpoint,model=self.model,optimizer=self.optimizer,epoch=epoch)
                logger.info(f"The last checkpoints was saved on [{self.config.save_path_checkpoint} ] for {epoch}.epoch")
                    
                metrics={"train_loss":train_loss,
                            "train_acc":train_acc,
                            "valid_loss": valid_loss,
                            "valid_acc":valid_acc,
                            "Epoch":epoch}
                    
                # save the metrics to the list
                result_list.append(metrics)

                # save the metrics to the mlflow
                mlflow.log_metrics(metrics=metrics,step=epoch)
                
            # Create an mlflow signature
            img_batch=next(iter(train_dataloader))[0].to(self.config.device)
            signature=infer_signature(model_input=img_batch.cpu().detach().numpy(),
                                          model_output=self.model(img_batch).cpu().detach().numpy())
                
            save_as_json(data=result_list,save_path=self.config.save_result_path)
            logger.info(f"Training results were saved as json file on [{self.config.save_result_path} ]")
                
            training_artifacts=TrainingArtifact(checkpoints_path=self.config.save_path_checkpoint,
                                                    results_path=self.config.save_result_path)

            # save final model
            save_obj(self.model,self.config.final_model_save_path)
            logger.info(f"Final model is saved on [{self.config.final_model_save_path}]")
        
            return training_artifacts,signature
        
        except Exception as e:
            ExceptionBlock(e,sys)
        
    def start_training_with_mlflow(self):
        
        try:
            
          # uri for mlflow track url in dagshub or local host
          
          # uri="https://dagshub.com/umitaktas/MLops_Project_-image_classification-.mlflow"   # for dagshub
            uri="0.0.0.0:5000"      # for local host
          
          # mlflow ui and other apps dont overlap
            subprocess.Popen(["mlflow","ui"])
            
            # MLFLOW tracking
            mlflow.set_tracking_uri(uri=uri)
            logger.info(f"MLflow was tracked on [{uri} ]")
            

            # create a new MLFLOW experiment
            # Son run ID'yi al
            
            mlflow.set_experiment("MLFLOW MyFirstExperiment")

            params={"Batch_size":self.config.batch_size,
                    "Learning_rate":self.config.lr,
                    "Betas":(self.config.beta1,self.config.beta2),
                    "Epoch":self.config.epoch}
            
            # start an MLFLOW run
            with mlflow.start_run():

                # log the hyperparameters (epoch,lr,vs)
                mlflow.log_params(params=params)
                
                # Set a tag that we can use to remind ourselves what this run was for
                mlflow.set_tag("Pytorch Training Info","Environment image classification training")
                
                # Training
                training_artifacts,signature=self.initiate_training()
                
                # log the model
                mlflow.pytorch.log_model(
                    pytorch_model=self.model,
                    artifact_path="FirstModel",
                    signature=signature,
                    registered_model_name="my_model_name")
                
                logger.info("Training is completed. Metrics, parameters and model was saved on MLflow")
                
            return training_artifacts
        
        except Exception as e:
            ExceptionBlock(e,sys)
            
                     
                        
            
        