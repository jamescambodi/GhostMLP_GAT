"""
This script describes a PyTorch class called `MultiviewPoint`, which loads 3D point cloud data from various perspectives.
The dataset supports shuffling, loading specific numbers of views per sample, and grouping data by class.

This script is intended to be used with the following dependencies:
- `numpy` for numerical operations.
 - `glob` for file path matching.
- `torch` and `torch.utils.data`  for PyTorch dataset utilities.
"""


import numpy as np  # For numerical operations
import glob  # For retrieving file paths matching a pattern
import torch.utils.data  # For defining custom PyTorch datasets
import torch  # For tensor operations

class MultiviewPoint(torch.utils.data.Dataset):
    """
    PyTorch dataset for loading multiview 3D point cloud data.

    Attributes:
        classnames (list): List of class names for the dataset.
        root_dir (str): Path to the root directory of the dataset.
        num_views (int): Number of views to load per sample.
        filepaths (list): List of file paths for the dataset files.
    """

    def __init__(self, root_dir, num_views=20, shuffle=True):
        """
        Initialize the dataset with root directory, number of views, and shuffle option.

        Args:
            root_dir (str): Path to the dataset root directory.
            num_views (int, optional): Number of views to load per sample. Defaults to 20.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        """
        # List of class names corresponding to the dataset's structure
        self.classnames = [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
            'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
            'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand',
            'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs',
            'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
        ]

        # Store root directory and number of views
        self.root_dir = root_dir
        self.num_views = num_views

        # Determine the set name (e.g., train, test) and parent directory
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]

        # Initialize filepaths to store paths to all files
        self.filepaths = []

        # Collect file paths for all classes
        for i in range(len(self.classnames)):
            # Retrieve all files for the class in the given dataset set (train/test/val)
            all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*.pth'))

            # Subsample files based on the number of views requested
            stride = int(20 / self.num_views)  # Adjust stride for sampling views
            all_files = all_files[::stride]  # Select every `stride`-th file

            # Append these file paths to the dataset's filepaths list
            self.filepaths.extend(all_files)

        # Shuffle the dataset if the shuffle option is enabled
        if shuffle:
            rand_idx = np.random.permutation(int(len(self.filepaths) / num_views))  # Randomize sample indices
            filepaths_new = []
            for i in range(len(rand_idx)):
                # Rearrange file paths based on the shuffled indices
                filepaths_new.extend(self.filepaths[rand_idx[i] * num_views:(rand_idx[i] + 1) * num_views])
            self.filepaths = filepaths_new

    def __len__(self):
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return int(len(self.filepaths) / self.num_views)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - class_id (int): The class ID of the sample.
                - all_point_set (torch.Tensor): A stacked tensor of point clouds from multiple views.
                - filepaths (list): List of file paths corresponding to the sample's views.
        """
        # Get the base path for the sample
        path = self.filepaths[idx * self.num_views]

        # Extract the class name and map it to a class ID
        class_name = path.split('/')[-3]  # Class name is the third last component in the path
        class_id = self.classnames.index(class_name)  # Get the class ID

        # Initialize a list to store the point sets for all views
        all_point_set = []

        # Load point clouds for all views
        for i in range(20):
            point_set = torch.load(
                self.filepaths[idx * self.num_views + i],  # Path to the current view's file
                weights_only=True,  # Load only weights if applicable
                map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Handle device compatibility
            )
            point_set = point_set.squeeze()  # Remove extra dimensions if present
            all_point_set.append(torch.tensor(point_set))  # Append the tensor to the list

        # Stack all views into a single tensor
        return (class_id, torch.stack(all_point_set), self.filepaths[idx * self.num_views:(idx + 1) * self.num_views])