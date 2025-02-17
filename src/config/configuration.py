from src.constants import PARAMS_FILE_PATH, CONFIG_FILE_PATH,SCHEMA_FILE_PATH
from src.utils import load_yaml
from src.entity.config_entity import *
import os


class Configuration():

    def __init__(self):

        self.params = load_yaml(PARAMS_FILE_PATH)
        self.config = load_yaml(CONFIG_FILE_PATH)
        self.schema= load_yaml(SCHEMA_FILE_PATH)

    def get_data_ingestion_configs(self):

        config = self.config.data_ingestion
        params = self.params.data_ingestion
        schema = self.schema.all

        data_ingetion_config = DataIngestionConfig(local_data_folder=config.local_data_folder,
                                                   labels=schema.labels,
                                                   save_train_img_path=os.path.join(config.save_dir, config.train_img_path_name),
                                                   save_test_img_path=os.path.join(config.save_dir, config.test_img_path_name),
                                                   save_valid_img_path=os.path.join(config.save_dir, config.valid_img_path_name),
                                                   valid_split_rate=params.valid_split_rate,
                                                   random_state=params.random_state)
        return data_ingetion_config

    def get_data_transformation_configs(self):

        config = self.config.data_transformation
        params = self.params.data_transformation

        data_transformation_config = DataTransformationConfig(save_path_train_dataset=os.path.join(config.save_dir, config.train_dataset_name),
                                                              save_path_valid_dataset=os.path.join(config.save_dir, config.valid_dataset_name),
                                                              save_path_test_dataset=os.path.join(config.save_dir, config.test_dataset_name),
                                                              resize_size=params.image_resize_size)
        return data_transformation_config

    def get_model_configs(self):

        config = self.config.model
        params = self.params.model

        model_config = ModelConfig(save_path_model=os.path.join(config.save_dir, config.model_name),
                                   channel_size=params.channel_size,
                                   img_size=params.img_size,
                                   label_size=params.label_size
                                   )
        return model_config

    def get_training_configs(self):

        config = self.config.train
        params = self.params.train

        training_config = TrainingConfig(save_path_checkpoint= os.path.join(config.checkpoint_dir, config.checkpoint_name),
                                         save_result_path= os.path.join(config.save_dir, config.result_name),
                                         batch_size=params.batch_size,
                                         beta1=params.beta1,
                                         beta2=params.beta2,
                                         lr=params.lr,
                                         epoch=params.epoch,
                                         device=params.device,
                                         load_checkpoints=params.load_checkpoints,
                                         final_model_save_path=config.final_model_save_path)
        return training_config

    def get_testing_configs(self):

        config = self.config.test
        params = self.params.test

        test_config = TestingConfig(save_result_path=os.path.join(config.save_dir, config.result_name),
                                    batch_size=params.batch_size,
                                    device=params.device,
                                    final_model_path=config.final_model_path,
                                    test_dataset_path=config.test_dataset_path,
                                    load_checkpoints=params.load_checkpoints ,
                                    save_tested_model=params.save_tested_model,
                                    tested_model_save_path=config.tested_model_save_path,
                                    checkpoints_path=config.checkpoint_path                               
                                    )
        return test_config

    def get_prediction_configs(self):

        config = self.config.prediction
        params = self.params.prediction
        schema = self.schema.all
        


        prediction_config = PredictionConfig(local_data_folder=config.local_data_folder,
                                             save_path_image_path=os.path.join(config.save_dir, config.image_path_name),
                                             save_path_result_path=os.path.join(config.save_dir, config.result_name),
                                             batch_size=params.batch_size,
                                             device=params.device,
                                             image_size=params.image_size,
                                             labels=schema.labels,
                                             final_model_path=config.final_model_path)
        return prediction_config
    

