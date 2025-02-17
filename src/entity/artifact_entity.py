from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionArtifact:
    train_data_dict_path: Path 
    valid_data_dict_path: Path
    test_data_dict_path: Path
    
@dataclass
class DataTransformationArtifact:
    train_dataset_paths: Path
    test_dataset_paths: Path
    validation_dataset_path: Path
    
@dataclass
class ModelArtifact:
    model_path: Path
    
@dataclass
class TrainingArtifact:
    checkpoints_path: Path
    results_path: Path
       

    