import pathlib
import torch
from pathlib import Path

def save_model(model,
               model_path : str,
               model_name : str):
  
  
  model_save_path = Path(model_path) / Path(model_name + ".pth")
  
  torch.save(model.state_dict(), f = model_save_path)
  print("model saved")

device = "cuda" if torch.cuda.is_available() else "cpu"

def return_loaded_model(base_model:torch.nn.Module,
               input_shape:int,
               output_shape :int,
               hidden_units:int,
               model_save_path : str,
               model:str):
  torch.manual_seed(42)

  model = base_model(input_shape = input_shape,
                     output_shape = output_shape,
                     hidden_units = hidden_units)
  
  model.load_state_dict(torch.load(f = model_save_path))


  return model.to(device)

################################################################################################
from torch.utils.tensorboard import SummaryWriter

# Create a writer with all default settings

def create_writer(experiment_name: str, 
                  model_name: str, 
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    """Creates a torch.utils.tensorboard.writer.SummaryWriter() instance saving to a specific log_dir.

    log_dir is a combination of runs/timestamp/experiment_name/model_name/extra.

    Where timestamp is the current date in YYYY-MM-DD format.

    Args:
        experiment_name (str): Name of experiment.
        model_name (str): Name of model.
        extra (str, optional): Anything extra to add to the directory. Defaults to None.

    Returns:
        torch.utils.tensorboard.writer.SummaryWriter(): Instance of a writer saving to log_dir.

    Example usage:
        # Create a writer saving to "runs/2022-06-04/data_10_percent/effnetb2/5_epochs/"
        writer = create_writer(experiment_name="data_10_percent",
                               model_name="effnetb2",
                               extra="5_epochs")
        # The above is the same as:
        writer = SummaryWriter(log_dir="runs/2022-06-04/data_10_percent/effnetb2/5_epochs/")
    """
    from datetime import datetime
    import os

    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
        
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)
