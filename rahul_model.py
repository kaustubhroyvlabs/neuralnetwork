
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, TensorDataset

from datetime import datetime as dt

flow_type = np.genfromtxt('../dataset2/FlowStructure_2022_03_24_total.dat', dtype=str)
vol_data = np.genfromtxt('../points_vol.dat', skip_header=1)
velocity_data = np.load('../dataset2/data.npy')
velocity_data_sliced = velocity_data[int(flow_type[0,0]):int(flow_type[-1,0])+1, :, :]

labels = np.unique(flow_type[:,1])
label2id = {k:v for k,v in enumerate(labels)}
id2label = {v:k for k,v in label2id.items()}


class ClassifierModel(nn.Module):

    #   Constructor Function:
    def __init__(self, num_classes, id2label, label2id) -> None:
        super().__init__()
        self.act = nn.GELU()
        self.mlp1 = nn.Linear(in_features=19875, out_features=9936)
        self.mlp2 = nn.Linear(in_features=9936, out_features=4968)
        self.mlp3 = nn.Linear(in_features=4968, out_features=1000)
        self.mlp4 = nn.Linear(in_features=1000, out_features=200)
        self.mlp5 = nn.Linear(in_features=200, out_features=50)
        self.mlp6 = nn.Linear(in_features=50, out_features=5)
        self.config = {}
        self.config['id2label']=id2label
        self.config['label2id']=label2id
        self.loss_fc = nn.CrossEntropyLoss(reduction='none')
        # p = self.mlp2.parameters()
        self.optimizer = None

    #   Call Function/Default Function:
    def forward(self, input):
        x = self.mlp1(input)
        x = self.act(x)
        x = self.mlp2(x)
        x = self.act(x)
        x = self.mlp3(x)
        x = self.act(x)
        x = self.mlp4(x)
        x = self.act(x)
        x = self.mlp5(x)
        x = self.act(x)
        x = self.mlp6(x)
        return x
    
    def predict(self, input):
        #   If you want to pass one time step: i.e. input.shape = 19875
        #   input = input.unsqueeze(0)
        #   input shape has to be (n, 19875)
        output = self(input)
        pred = output.argmax(dim=-1)
        return pred
    
    def predict_classes(self, input):
        pred = self.predict(input)
        return [self.config['id2label'][str(i)] for i in pred]

    def create_optimizer(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad==True]
        self.optimizer = torch.optim.Adam(params=trainable_params, lr = 5e-3)

    def train_one_epoch(self, data, target, test_data, test_targets):
        if self.optimizer is None:
            self.create_optimizer()
        """
        In general in batched training, train_one_epoch will call training_step function
        which processes one batch of data at a time
        data(Dataset/Dataloader) -> shape = (n_batches, batch_size, input_tensor.shape)
        target(Dataset/Dataloader) -> shape = (n_batches, batch_size, label.shape)
        for i in range(n_batches):
            training_step(data[i], targets[i])

        if metric is not None:
            dict_metric = {
                "accuracy": self.compute_accuracy,
                "f1": self.prec_rec,
                "confusion": self.confusion_matrix,
            }
            a = dict_metric[metric](x_test, y_test)
            print(f"{metric}:"a)
        

        """
        for idx in range(len(data)):
            #print(idx)
            input = data[idx].unsqueeze(0)
            output = self(input)
            label = target[idx].unsqueeze(0)
            loss = self.loss_fc(output, label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        accuracy = self.compute_accuracy(test_data, test_targets)
        confusion_matrix = self.confusion_matrix(test_data, test_targets)
        other_metrics = self.f1_matrix(confusion_matrix)
        #precision_matrix = other_metrics[0]
        #recall_matrix = other_metrics[1]
        f1_matrix = other_metrics[2]
        
        print("Accuracy: %f \n" %accuracy)
        print("Confusion Matrix: \n" %confusion_matrix)
        print("f1 matrix: \n" %f1_matrix)
        #print("Precision: \n" %precision_matrix)
        #print("Recall: \n" %recall_matrix)

    def training_step(self, one_batch_of_data, one_batch_of_target):
        output = self(one_batch_of_data)
        label = one_batch_of_target
        loss = self.loss_fc(output, label)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        """...blah blah blah..."""
        
    def fit(self, data, targets, test_data, test_targets, epochs):
        for epoch in range(epochs):
            print(dt.now())
            print("Currently training epoch %s out of %s" %(epoch, epochs))
            self.train_one_epoch(data, targets, test_data, test_targets)

    def compute_accuracy(self, data, target):
        print("compute_accuracy start")
        acc = 0
        for idx in range(len(data)):
            #print("Accuracy %d" %idx)
            input = data[idx].unsqueeze(0)
            output = self(input).squeeze(0)
            pred = output.argmax(dim=-1)
            label = target[idx]
            if pred==label:
                acc+=1
        return float(acc/len(data))

    def confusion_matrix(self, data, target):
        print("confusion_matrix start")
        conf_mat = torch.zeros(size=(target.max()+1, target.max()+1))
        for idx in range(len(data)):
            #print("Confusion %d" %idx)
            input = data[idx].unsqueeze(0)
            output = self(input).squeeze(0)
            pred = output.argmax(dim=-1)
            label = target[idx]
            conf_mat[label][pred]+=1
        return conf_mat
    
    def f1_matrix(self, conf_mat):
        p_matrix = torch.zeros(len(conf_mat))
        r_matrix = torch.zeros(len(conf_mat))
        for i in range(len(conf_mat)):
            p_matrix[i] = conf_mat[i][i]/torch.sum(conf_mat[i])
            r_matrix[i] = conf_mat[i][i]/torch.sum(conf_mat[:,i])
            
        f1_matrix = (p_matrix*r_matrix)/(p_matrix+r_matrix)
        return [p_matrix, r_matrix, f1_matrix]

    def compile(self, optimizer, loss, metric):
        """
        Optimizer:  Accepts a object of the optimizer class or a string value that corresponds to a default optimizer object
                        ex.: torch.optim.Adam, "adam", "adagrad" etc.

        metric:     Accepts a string either "accuracy", "f1", "confusion"
        """
        trainable_params = [p for p in self.parameters() if p.requires_grad==True]
        if type(optimizer)==str:
            dict_optim = {
                "adam": torch.optim.Adam,
                "adagrad": torch.optim.Adagrad,
                "sgd": torch.optim.SGD,
            }
            self.optimizer = dict_optim[optimizer](trainable_params)
        else:
            self.optimizer = optimizer(trainable_params)
        
        if type(loss)==str:
            dict_loss = {
                "crossentropy": torch.nn.CrossEntropyLoss(),
                "mseloss": torch.nn.MSELoss(),
            }
            self.loss_fc=dict_loss[loss]
        else:
            self.loss_fc=loss
            
        if metric is not None:
            dict_metric = {
                "accuracy": self.compute_accuracy,
                "f1": self.f1_matrix,
                "confusion": self.confusion_matrix,
            }
            verbose = dict_metric[metric](x_test, y_test)
            print(f"{metric}:",verbose)
            
        self.metric = metric
                
data = torch.tensor(velocity_data_sliced[1800:9000], dtype = torch.float32).reshape(7200, 19875)
targets = torch.tensor([id2label[i] for i in flow_type[1800:9000, 1]])

test_data = np.vstack((velocity_data_sliced[0:1800], velocity_data_sliced[9000:10800]))
test_data = torch.tensor(test_data, dtype = torch.float32).reshape(3600, 19875)

test_targets1 = [id2label[i] for i in flow_type[0:1800, 1]]
test_targets2 = [id2label[i] for i in flow_type[9000:10800, 1]]
test_targets = np.append(test_targets1, test_targets2)
test_targets = torch.tensor(test_targets)

dataset_normal = CustomTensorDataset(tensors=(data, targets), transform=None)
dataset_loader = torch.utils.data.DataLoader(dataset_normal, batch_size=10)

test_dataset_normal = CustomTensorDataset(tensors=(test_data, test_targets), transform=None)
test_dataset_loader = torch.utils.data.DataLoader(test_dataset_normal, batch_size=10)

num_classes = targets.max()+1
model = ClassifierModel(num_classes, id2label, label2id)
model.fit(data, targets, test_data, test_targets, epochs=10)
torch.save(model,"initial_trained_model.pt")
print(dt.now())