import torch
import tqdm
from src.exception.exception import ExceptionBlock,sys


    
def model_validation(valid_dataloader, loss_fn, Model, device):
    try:  
        Model.eval()
        
        progress_bar = tqdm.tqdm(range(len(valid_dataloader)), "Validation Process")
        
        with torch.no_grad():

            valid_loss_value = 0
            valid_correct_value = 0
            valid_total_value = 0

            for batch_valid, (img, label) in enumerate(valid_dataloader):

                img_valid = img.to(device)
                label_valid = label.to(device)

                out_valid = Model(img_valid)
                loss_valid = loss_fn(out_valid, label_valid)
                valid_loss_value += loss_valid.item()

                _, predictions_valid = torch.max(out_valid, 1)

                valid_correct_value += (predictions_valid == label_valid).sum().item()
                valid_total_value += label_valid.size(0)
                progress_bar.update(1)

        total_loss = valid_loss_value/(batch_valid+1)
        total_acc = (valid_correct_value/valid_total_value)*100
        
        progress_bar.set_postfix({"valid_acc":total_acc,
                                  "valid_loss":total_loss})

        return total_loss, total_acc
    
    except Exception as e:
        raise ExceptionBlock(e,sys)