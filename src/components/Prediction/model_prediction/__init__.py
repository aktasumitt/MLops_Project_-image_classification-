import torch
from src.exception.exception import ExceptionBlock,sys


def predict(prediction_dataloader, Model, batch_size, devices="cpu"):
    try:
        prediction_list=[]
        excess_len=0
        for _b, (img_test, _l) in enumerate(prediction_dataloader):
            
            # control batch_size
            if img_test.shape[0]!=batch_size:
                excess_len=batch_size-(len(img_test)%batch_size)
                C,H,W=img_test[0].shape
                add_data=torch.zeros((excess_len,C,H,W))
                img_test=torch.cat([img_test,add_data],dim=0)

            img_test = img_test.to(devices)
            
            out_test = Model(img_test)    

            _, predictions_valid = torch.max(out_test, 1)
            if excess_len!=0:
                predictions_valid=predictions_valid[:-excess_len]
            
            prediction_list.append(predictions_valid)
        
        return torch.stack(prediction_list).reshape(-1).tolist()
    
    except Exception as e:
        raise ExceptionBlock(e,sys)