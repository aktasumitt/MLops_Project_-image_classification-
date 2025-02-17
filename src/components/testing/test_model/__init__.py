import torch
import tqdm
from src.exception.exception import ExceptionBlock,sys

    
def model_testing(test_dataloader, loss_fn, Model, devices="cpu"):
    try:    
        progress_bar = tqdm.tqdm(range(len(test_dataloader)), "Test Process")
        
        with torch.no_grad():

            test_loss_value = 0
            test_correct_value = 0
            test_total_value = 0

            for batch_test, (img, label) in enumerate(test_dataloader):

                img_test = img.to(devices)
                label_test = label.to(devices)

                out_test = Model(img_test)
                loss_test = loss_fn(out_test, label_test)
                test_loss_value += loss_test.item()

                _, predictions_valid = torch.max(out_test, 1)

                test_correct_value += (predictions_valid == label_test).sum().item()
                test_total_value += label_test.size(0)
                progress_bar.update(1)

        total_loss = test_loss_value/(batch_test+1)
        total_acc = (test_correct_value/test_total_value)*100
        
        progress_bar.set_postfix({"valid_acc":total_acc,
                                  "valid_loss":total_loss})

        return total_loss, total_acc
    
    except Exception as e:
        raise ExceptionBlock(e,sys)