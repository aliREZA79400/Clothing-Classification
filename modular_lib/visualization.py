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
