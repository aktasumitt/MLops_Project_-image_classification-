from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    local_data_folder: Path
    save_train_img_path: Path
    save_test_img_path: Path
    save_valid_img_path: Path
    labels: dict
    valid_split_rate: float
    random_state: int


@dataclass
class DataTransformationConfig:
    save_path_train_dataset: Path
    save_path_valid_dataset: Path
    save_path_test_dataset: Path
    resize_size: int


@dataclass
class ModelConfig:
    save_path_model: Path
    channel_size: int
    img_size: int
    label_size: int


@dataclass
class TrainingConfig:
    save_path_checkpoint: Path
    save_result_path: Path
    batch_size: int
    beta1: float
    beta2: float
    lr: float
    epoch: int
    device: str
    load_checkpoints: bool
    final_model_save_path: Path


@dataclass
class TestingConfig:
    test_dataset_path: Path
    save_result_path: Path
    batch_size: int
    device: str
    final_model_path: Path
    load_checkpoints: bool
    checkpoints_path: Path
    save_tested_model: bool
    tested_model_save_path: Path = None


@dataclass
class PredictionConfig:
    local_data_folder: Path
    save_path_image_path: Path
    save_path_result_path: Path
    labels: dict
    batch_size: int
    device: str
    image_size: int
    final_model_path: Path
