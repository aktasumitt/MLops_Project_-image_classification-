import torch
import tqdm
from src.exception.exception import ExceptionBlock,sys


def model_training(train_dataloader, optimizer, loss_fn, Model,device):
    try: 
        Model.train()
        
        train_loss_value = 0
        train_correct_value = 0
        train_total_value = 0

        progress_bar = tqdm.tqdm(range(len(train_dataloader)), "Training Process")

        for batch_train, (img, label) in enumerate(train_dataloader):

            img_train = img.to(device)
            label_train = label.to(device)
            
            optimizer.zero_grad()
            out_train = Model(img_train)
            loss_train = loss_fn(out_train, label_train)

            train_loss_value += loss_train.item()

            _, predictions_train = torch.max(out_train, 1)
            
            loss_train.backward()
            optimizer.step()

            train_correct_value += (predictions_train == label_train).sum().item()
            train_total_value += label_train.size(0)

            progress_bar.update(1)
        
        
        total_loss = train_loss_value/(batch_train+1)
        total_acc = (train_correct_value/train_total_value)*100
        
        progress_bar.set_postfix({"train_acc":total_acc,
                                  "train_loss":total_loss})
                
        return total_loss, total_acc
   
    except Exception as e:
            raise ExceptionBlock(e,sys)
    

        