from torchvision.transforms import transforms
from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
from src.exception.exception import ExceptionBlock,sys


class Create_Dataset(Dataset):
    def __init__(self,dataset_path_dict:dict,transformer:transforms):
        super().__init__()
        self.transformer=transformer
        self.dataset_paths_dict=dataset_path_dict
    
    def __len__(self):
            return len(list(self.dataset_paths_dict.items())) 
        
    def __getitem__(self, index):
        try:
            image_paths_label_list=list(self.dataset_paths_dict.items())
            image_path_label=image_paths_label_list[index]
            img_path=image_path_label[0]
            label=image_path_label[1]
            
            img=Image.open(img_path).convert("RGB")
            transformed_img=self.transformer(img)
            
            return (transformed_img,label)
        
        except Exception as e:
            raise ExceptionBlock(e,sys)
