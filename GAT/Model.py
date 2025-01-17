"""
This script outlines a class called `Model` designed for PyTorch models to offer functions and features for their implementation.  
Creating backups of models and reloading them is essential. It's also helpful to have a framework that allows for expansion.  
The class is designed to simplify model management 
during training and evaluation.
"""

import torch
import torch.nn as nn
import os
import glob

class Model(nn.Module):
    """
    Creating a PyTorch model class with functions for saving and loading. 
    Overseeing the maintenance of model checkpoints.

    Attributes:
        name (str): The name of the model, used for organizing saved checkpoints.
    """

    def __init__(self, name):
        """
        Initialise the base model with a name.

        Args:
            name (str): The name of the model for checkpoint organization.
        """
        super(Model, self).__init__()
        self.name = name

    def save(self, path, epoch=0):
        """
        Save the model's state dictionary to a specified directory.

        Args:
            path (str): The base path where the model's checkpoints will be saved.
            epoch (int, optional): The current epoch number (not used in the default implementation).
        """
        complete_path = os.path.join(path, self.name)  # Create model-specific directory
        if not os.path.exists(complete_path):  # Ensure the directory exists
            os.makedirs(complete_path)
        # Save the model's state dictionary
        torch.save(self.state_dict(),
                   os.path.join(complete_path, "model.pth"))
        print(f"Model saved to {os.path.join(complete_path, 'model.pth')}")

    def save_results(self, path, data):
        """
        Placeholder for saving additional results (to be implemented in subclasses).

        Args:
            path (str): The path to save results.
            data (object): The results data to be saved.

        Raises:
            NotImplementedError: This method must be implemented in subclasses.
        """
        raise NotImplementedError("Model subclass must implement this method.")

    def load(self, path, modelfile=None):
        """
        Load the model's state from a checkpoint file.

        Args:
            path (str): The base path where the model's checkpoints are stored.
            modelfile (str, optional): The specific checkpoint file to load. If not provided, 
                                       the latest checkpoint in the directory is loaded.

        Raises:
            IOError: If the model directory does not exist.
        """
        complete_path = os.path.join(path, self.name)  # Model-specific directory
        if not os.path.exists(complete_path):  # Check if the directory exists
            raise IOError(f"{self.name} directory does not exist in {path}")

        # Determine which checkpoint file to load
        if modelfile is None:
            model_files = glob.glob(os.path.join(complete_path, "*"))
            if not model_files:  # Ensure there are files to load
                raise IOError(f"No checkpoint files found in {complete_path}")
            mf = max(model_files, key=os.path.getctime)  # Select the latest file
        else:
            mf = os.path.join(complete_path, modelfile)

        # Load the state dictionary into the model
        self.load_state_dict(torch.load(mf))
        print(f"Model loaded from {mf}")
