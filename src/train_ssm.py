""" 
Entry point for training the State Space Model


This model will
- Instantiate the VAE from models.py
- Instantiate the SSM from models.py (Stochastic or Deterministic or Recurrent)
- Use the data loader to get data
    - Data should be compressed using the VAE before training
"""
from config import GlobalConfig, SSMConfig
from pathlib import Path
import argparse
import os
import glob
import time


def train(args):

    # Make new sub folder for this particular run
    this_time_folder = os.path.join(args.model_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
    Path(this_time_folder).mkdir(parents=True, exist_ok=True)

    if args.model_in_folder:
        this_time_folder = args.model_in_folder
        joined_path = os.path.join(args.model_in_folder, "*_generator_model.h5")
        poss_files = glob.glob(joined_path)
        latest_i = max([int(os.path.split(x)[1].split("_")[0]) for x in poss_files])

        #TODO: Load model weights from existing
        offset = latest_i
    else:
        offset = 0


    #TODO: Main training loop

    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", required=False, default=SSMConfig.NUM_EPOCHS,
                        help="Number of epochs to train.", type=int)
    parser.add_argument("--learning_rate", required=False, default=SSMConfig.LEARNING_RATE,
                        help="Learning rate")
    parser.add_argument("--hidden_dim", required=False, default=SSMConfig.HIDDEN_DIM,
                        help="The number of elements in hidden dimension")
    parser.add_argument("--model_dir", required=False,
                        default=GlobalConfig.get("MODEL_DIR"))
    parser.add_argument("--batch_size", required=False, type=int,
                        default=SSMConfig.BATCH_SIZE)
    parser.add_argument("--model_in_folder", required=False,
        help="Path to a folder containing disc / generator weights. The latest one will be loaded.")
    args = parser.parse_args()

    train(args)