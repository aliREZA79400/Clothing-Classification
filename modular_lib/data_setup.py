import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from pathlib import Path
import torch

def creat_dataloaders(train_path : None,
                      test_path : None,
                      val_path : None,
                      batch_size : int,
                      train_transform : transforms,
                      test_transform  : transforms,
                      seed : int):
    """This function build dataloaders


    Args:
    train_path : path of train data
    test_path : path of test data
    val_path : path of val data
    batch_size : number of data samples in each batch
    train_transform : data transform for train data
    test_transform : data transform for test data
    seed : for reproducibility of results

    Return:
    \ntrain_dataloader , test_dataloader , val_dataloader , class_names , class_idx
    """
    torch.manual_seed(seed = seed)
    torch.cuda.manual_seed(seed=seed)
    
    train_data = datasets.ImageFolder(root=train_path,
                                    transform=train_transform)

    test_data = datasets.ImageFolder(root=test_path,
                                        transform=test_transform)
    if val_path  is not None :
        val_data = datasets.ImageFolder(root=val_path,
                                        transform=test_transform)
        val_dataloader = DataLoader(val_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=os.cpu_count(),
                                pin_memory=True,
                                drop_last=True)
        

    train_dataloader = DataLoader(train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=os.cpu_count(),
                                pin_memory=True,
                                drop_last=True)

    test_dataloader = DataLoader(test_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=os.cpu_count(),
                                pin_memory=True,
                                drop_last=True)
    
    class_names = train_data.classes
    class_idx = train_data.class_to_idx

    if val_dataloader :
        return train_dataloader , test_dataloader , val_dataloader , class_names , class_idx
    else:
        return train_dataloader , test_dataloader , class_names , class_idx

