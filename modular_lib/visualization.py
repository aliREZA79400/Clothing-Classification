import torch
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

def pred_image_label(model : torch.nn.Module,
                     transform : transforms,
                     dataset_path : None,
                     class_name : list):
  
  image_path_list = list(dataset_path.glob("*/*.jpg"))
  random_image_path = random.choice(image_path_list)
  
  custom_image_ = (torchvision.io.read_image(str(random_image_path)) / 255.).to(device)

  custom_image_1 = transform(custom_image_).unsqueeze(dim = 0)

  y_pred_logit = model(custom_image_1)

  y_prob_label =  torch.softmax(y_pred_logit ,dim = 1)

  prob = torch.max(y_prob_label)

  y_label = torch.argmax(y_prob_label , dim = 1)
  print(y_label)
  plt.figure(figsize = (10,6))

  plt.imshow(custom_image_.squeeze(dim=0).permute(1,2,0).cpu())

  title = class_name[y_label]

  plt.title(f"{title} | probability : {str(prob.cpu().detach().numpy())[:4]} ")
######################################################################################
from PIL import Image
from pathlib import Path

def original_random_image_show(data_path,
                             seed:int = 42,
                             rows:int=3,
                             cols:int=3):
    """This function show nrows*ncols image from original dataset with their shapes
    """
    random.seed(seed)

    data_path = Path(data_path)

    image_path_list = list(data_path.glob("*/*/*.jpg"))


    rows = rows
    cols = cols
    fig  = plt.figure(figsize=(16,14))

    for i in range(1 , rows * cols +1):
        fig.add_subplot(rows,cols,i)
        image_random_path = random.choice(image_path_list)
    
        im_read = plt.imread(image_random_path)

        title = f" {image_random_path.parent.stem} | {im_read.shape} " 
        plt.imshow(im_read)
        plt.axis(False)
        plt.title(title)

######################################################################################


def visulize_transfomed_images(dataset : torchvision.datasets ,class_names :list , rows:int , cols:int):
    """This function show nrows*ncols image from transformed datasets with their labels
    """

    fig = plt.figure(figsize=(10,6))
    rows =rows
    cols = cols
    for i in range(1 ,rows*cols +1):
        random_image_idx = random.choice(list(range(len(dataset))))
        
        fig.add_subplot(rows,cols,i)

        plt.imshow(dataset[random_image_idx][0].permute(1,2,0));
        plt.axis(False)
        plt.title(class_names[dataset[random_image_idx][1]])


