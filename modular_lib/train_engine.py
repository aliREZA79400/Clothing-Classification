from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

def train_model(
        model : torch.nn.Module ,
        train_dataloader : DataLoader,
        test_dataloader : DataLoader,
        loss_function : torch.nn,
        optimizer : torch.optim,
        epoch_number : int ,
        metric_function : None,
        device : None,
        writer : SummaryWriter):
    

    ### save the loss and accuracy per epoch
    train_loss_list = []
    train_acc_list = []
    test_acc_list = []
    test_loss_list = []


    for epochs in tqdm(range(epoch_number)):

        train_loss = 0
        train_acc = 0
        test_loss = 0
        test_acc = 0

        print(f"Number of Epoch : {epochs + 1}")

        ### train step of train model
        for batch , (X ,y) in enumerate(train_dataloader):

            X , y = X.to(device) , y.to(device)

            model.train()

            y_preds = model(X)

            loss = loss_function(y_preds , y)
            
            ### for syncing the devices
            loss_ = loss.cpu().detach().numpy()

            train_loss += loss_

            y_preds_label = torch.argmax(torch.softmax(y_preds,dim=1),dim=1)

            acc = metric_function(y.cpu(),y_preds_label.cpu())

            train_acc += acc

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        ### Avarege loss and acc of train
        train_loss = train_loss / len(train_dataloader)
        train_acc = train_acc / len(train_dataloader)
        ### save data in list
        train_loss_list.append(train_loss)
        train_acc_list.append(train_acc)


        ### test step of train model
        model.eval()
        with torch.inference_mode():
            for batch , (X , y) in enumerate(test_dataloader):
                X , y = X.to(device) , y.to(device)

                y_preds_test = model(X)

                loss_test = loss_function(y_preds_test , y)
                
                ### for syncing the devices
                loss_test_ = loss_test.cpu().detach().numpy()
                test_loss += loss_test_

                ### calculate accuracy
                y_preds_label_test = torch.argmax(torch.softmax(y_preds_test,dim=1),dim=1)

                acc = metric_function(y.cpu(),y_preds_label_test.cpu())

                test_acc += acc
        ### avarage of loss an acc in test step
        test_acc = test_acc / len(test_dataloader)
        test_loss = test_loss / len(test_dataloader)
        
        ### save data in test step
        test_acc_list.append(test_acc)
        test_loss_list.append(test_loss)

        print(f"train loss : {train_loss} | train acc : {train_acc} | test loss : {test_loss} | test acc : {test_acc}")
        
        writer.add_scalars(main_tag = "Loss",
                      tag_scalar_dict ={
                          "train_loss" : train_loss,
                          "test_loss" :test_loss
                      },global_step=epochs)
    
        writer.add_scalars(main_tag = "Accuracy",
                    tag_scalar_dict ={
                        "train_acc" : train_acc,
                        "test_acc" :test_acc
                    },global_step=epochs)
        
        writer.add_graph(model=model,
                         input_to_model=torch.randn(32,3,224,224).to(device))
        
    writer.close()
    model_results = {
        "train_loss" : train_loss_list,
        "train_accuracy" : train_acc_list,
        "test_loss" : test_loss_list,
        "test_accuracy" : test_acc_list
    }

    return model_results
