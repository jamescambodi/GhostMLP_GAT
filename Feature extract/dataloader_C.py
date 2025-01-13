import numpy as np
import glob
import torch.utils.data
import torch

class MultiviewPoint(torch.utils.data.Dataset):
    def __init__(self, root_dir, num_views=20, shuffle=True):
        self.classnames = ['bed', 'chair', 'desk']

        self.root_dir = root_dir
        self.num_views = num_views
        set_ = root_dir.split('/')[-1]
        parent_dir = root_dir.rsplit('/', 2)[0]
        self.filepaths = []

        for i in range(len(self.classnames)):
            all_files = sorted(glob.glob(parent_dir + '/' + self.classnames[i] + '/' + set_ + '/*.pth'))
            stride = int(20 / self.num_views)
            all_files = all_files[::stride]
            self.filepaths.extend(all_files)


        if shuffle == True:
            rand_idx = np.random.permutation(int(len(self.filepaths) / num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.filepaths[rand_idx[i] * num_views:(rand_idx[i] + 1) * num_views])
            self.filepaths = filepaths_new

    def __len__(self):
        return int(len(self.filepaths) / self.num_views)

    def __getitem__(self, idx):
        # Load the 20 views for the current point cloud
        path = self.filepaths[idx * self.num_views]
        class_name = path.split('/')[-2]
        class_id = self.classnames.index(class_name)
        class_id = torch.tensor(class_id, dtype=torch.long)  # Ensure label is a scalar Long tensor

        all_point_set = []
        for i in range(self.num_views):  # Load all views
            point_set = torch.load(self.filepaths[idx * self.num_views + i])
            point_set = point_set.squeeze()
            all_point_set.append(torch.tensor(point_set, dtype=torch.float32))  # Ensure features are Float

        # Stack the views into a tensor of shape [num_views, feature_dim]
        return torch.stack(all_point_set), class_id
