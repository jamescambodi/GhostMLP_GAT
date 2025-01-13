"""
This script sets up a training process for a model that includes loading data. 
In PyTorch, training and evaluation processes and the ability to process and view data efficiently are supported and saved the trained model.  
The system monitors performance indicators with TensorBoard and includes elements such as the Trainer.  
A program to oversee training and validation tasks with a setup for customisation. 
Setting the parameters for training.
"""

import numpy as np
import random
import torch
import torch.optim as optim
import torch.nn as nn
import os
import argparse
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import math
from PintView_GCN import PointViewGCN  # Custom model class for PointViewGCN
from dataloader import MultiviewPoint  # Custom dataloader for multi-view datasets

def seed_torch(seed=9990):
    """
    Set seed for reproducibility across different libraries and environments.
    
    Args:
        seed (int): Seed value for random number generators.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class Trainer(object):
    """
    Handles the training and evaluation of the GCN on GhostMLP features.
    
    Attributes:
        optimizer (torch.optim): Optimizer for model training.
        model (torch.nn.Module): The PointViewGCN model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        loss_fn (callable): Loss function used during training.
        num_views (int): Number of views per sample in the dataset.
        log_dir (str): Directory for logging training metrics.
        writer (SummaryWriter): TensorBoard writer for logging metrics.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, log_dir, num_views=20):
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.num_views = num_views
        self.model.cuda()  # Move model to GPU
        self.log_dir = log_dir
        self.writer = SummaryWriter()  # Initialize TensorBoard writer

    def train(self, n_epochs):
        """
        Perform training and validation over 100 epochs.
        
        Args:
            n_epochs (int): Number of epochs to train the model.
        """
        global lr
        best_acc = 0
        i_acc = 0
        self.model.train()  # Set model to training mode
        for epoch in range(n_epochs):
            if epoch == 1:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                if epoch > 1:
                    # Adjust learning rate using cosine annealing
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.5 * (1 + math.cos(epoch * math.pi / 15))

            # Shuffle and update training filepaths
            rand_idx = np.random.permutation(int(len(self.train_loader.dataset.filepaths) / self.num_views))
            filepaths_new = []
            for i in range(len(rand_idx)):
                filepaths_new.extend(self.train_loader.dataset.filepaths[
                                     rand_idx[i] * self.num_views:(rand_idx[i] + 1) * self.num_views])
            self.train_loader.dataset.filepaths = filepaths_new

            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            for i, data in enumerate(self.train_loader):
                if epoch == 0:
                    for param_group in self.optimizer.param_groups:
                        # Gradual warm-up of learning rate
                        param_group['lr'] = lr * ((i + 1) / (len(rand_idx) // 20))

                B, V, C = data[1].size()  # Extract batch size, views, and feature dimensions
                in_data = Variable(data[1]).view(-1, C)  # Flatten input views
                target = Variable(data[0]).cuda().long()  # Move target to GPU and convert to long
                target_ = target.unsqueeze(1).repeat(1, 4 * (5 + 10 + 15)).view(-1)  # Repeat for auxiliary loss
                self.optimizer.zero_grad()  # Reset gradients
                out_data, F_score, F_score_m, F_score2 = self.model(in_data)  # Forward pass
                out_data_ = torch.cat((F_score, F_score_m, F_score2), 1).view(-1, 40)  # Auxiliary outputs
                loss = self.loss_fn(out_data, target) + self.loss_fn(out_data_, target_)  # Compute loss

                # Log training metrics
                self.writer.add_scalar('train/train_loss', loss, i_acc + i + 1)
                pred = torch.max(out_data, 1)[1]  # Get predicted labels
                results = pred == target  # Compare predictions with targets
                correct_points = torch.sum(results.long())  # Count correct predictions
                acc = correct_points.float() / results.size()[0]  # Compute accuracy
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc + i + 1)
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights

                # Print training progress
                log_str = f'epoch {epoch + 1}, step {i + 1}: train_loss {loss:.3f}; train_acc {acc:.3f}'
                if (i + 1) % 1 == 0:
                    print(log_str)
            i_acc += i

            # Perform validation every epoch
            if (epoch + 1) % 1 == 0:
                with torch.no_grad():
                    val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch + 1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch + 1)
                # Save the best model
                if val_overall_acc > best_acc:
                    best_acc = val_overall_acc
                    print('the best model_is saving')
                    self.model.save(self.log_dir + '/')
                print('best_acc', best_acc)
        self.writer.close()  # Close TensorBoard writer

    def update_validation_accuracy(self, epoch):
        """
        Evaluate the model on the validation dataset and compute metrics.
        
        Args:
            epoch (int): epoch number.

        Returns:
            tuple: Overall validation accuracy and mean class accuracy.
        """
        all_correct_points = 0
        all_points = 0
        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        self.model.eval()  # Set model to evaluation mode
        targ_numpy, pred_numpy = [], []
        for _, data in enumerate(self.val_loader, 0):
            B, V, C = data[1].size()
            in_data = Variable(data[1]).view(-1, C)
            target = Variable(data[0]).cuda()
            out_data, F_score, F_score_m, F_score2 = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            targ_numpy.append(np.asarray(target.cpu()))
            pred_numpy.append(np.asarray(pred.cpu()))
            results = pred == target

            # Update class-wise statistics
            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())
            all_correct_points += correct_points
            all_points += results.size()[0]

        # Compute metrics
        print('Total # of test models: ', all_points)
        class_acc = (samples_class - wrong_class) / samples_class  # Class-wise accuracy
        val_mean_class_acc = np.mean(class_acc)  # Mean class-wise accuracy
        acc = all_correct_points.float() / all_points  # Overall accuracy
        val_overall_acc = acc.cpu().data.numpy()
        print('val mean class acc. : ', val_mean_class_acc)
        print('val overall acc. : ', val_overall_acc)
        print(class_acc)
        self.model.train()  # Reset model to training mode
        return val_overall_acc, val_mean_class_acc


# Argument parser configuration
parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="path of the model", default="log")
parser.add_argument("-bs", "--batchSize", type=int, default=128)
parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate in training [default: 0.0001]')
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.09)
parser.add_argument("-num_views", type=int, help="number of views", default=20)
parser.add_argument('--train_path', default='../data/modelnet_trained_feature/*/train', help='path of the trained feature for train data')
parser.add_argument('--val_path', default='../data/modelnet_trained_feature/*/test', help='path of the trained feature for test data')
parser.add_argument("--workers", default=0)
parser.set_defaults(train=False)

if __name__ == '__main__':
    """
    Main entry point.
    Initialises datasets, dataloaders, model, optimizer, and trainer, and starts training.
    """
    seed_torch()  # Set random seed
    args, unknown = parser.parse_known_args()
    log_dir = args.name
    cnet_2 = PointViewGCN(args.name, nclasses=40, num_views=args.num_views)  # Initialize the model
    optimizer = optim.Adam(cnet_2.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)  # Optimizer
    train_dataset = MultiviewPoint(args.train_path)  # Load training dataset
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batchSize, shuffle=False, num_workers=args.workers)  # Training DataLoader
    val_dataset = MultiviewPoint(args.val_path)  # Load validation dataset
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=20, shuffle=False, num_workers=args.workers)  # Validation DataLoader
    print('num_train_files: ' + str(len(train_dataset.filepaths)))
    print('num_val_files: ' + str(len(val_dataset.filepaths)))
    trainer = Trainer(cnet_2, train_loader, val_loader, optimizer, nn.CrossEntropyLoss(), log_dir, num_views=args.num_views)  # Trainer instance
    #cnet_2.load_state_dict(torch.load('log/model_GCN_PointNet.pth', weights_only=True))
    #trainer.update_validation_accuracy(1)
    trainer.train(100)