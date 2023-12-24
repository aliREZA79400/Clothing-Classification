import torchvision
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import os

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

###################################################################
from typing import Tuple, Dict, List

# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
################################################################
from itertools import chain

def make_smaller(dir , persent , show:bool=False):
    random_path = []
    for root , dirs , files in os.walk(dir):
        if show == True:
            print(f"Root : {Path(root).stem} | Dirs : {len(dirs)} | files : {len(files)}")
        
        random_files = random.choices(files,k = (len(files) * persent) // 100)

        random_path.append([Path(root)/i for i in random_files])
    
    random_path =list(chain(*random_path))

    return random_path
################################################################
# Write a custom dataset class (inherits from torch.utils.data.Dataset)
from torch.utils.data import Dataset
from PIL import Image
# 1. Subclass torch.utils.data.Dataset
class ImageFolderCustom(Dataset):
    
    # 2. Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None,persent:int=20) -> None:
        
        # 3. Create class attributes
        # Get all image paths
        self.paths = make_smaller(targ_dir,persent=persent) # note: you'd have to update this if you've got .png's or .jpeg's
        # Setup transforms
        self.transform = transform
        # Create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir)

    # 4. Make function to load images
    def load_image(self, index: int) -> Image.Image:
        "Opens an image via a path and returns it."
        image_path = self.paths[index]
        return Image.open(image_path) 
    
    # 5. Overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Dataset)
    def __len__(self) -> int:
        "Returns the total number of samples."
        return len(self.paths)
    
    # 6. Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Dataset)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label (X, y)."
        img = self.load_image(index)
        class_name  = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name]

        # Transform if necessary
        if self.transform:
            return self.transform(img), class_idx # return data, label (X, y)
        else:
            return img, class_idx # return data, label (X, y)

#######################################################################################

def creat_dataloaders_v2(train_path : None,
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
    
    train_data = ImageFolderCustom(targ_dir=train_path,
                                    transform=train_transform)

    test_data = ImageFolderCustom(targ_dir=test_path,
                                        transform=test_transform)
    if val_path  is not None :
        val_data = ImageFolderCustom(targ_dir=val_path,
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

