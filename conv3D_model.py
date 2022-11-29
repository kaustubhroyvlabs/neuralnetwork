#%%
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
#%%
flow_type = np.genfromtxt('FlowStructure_2022_03_24_total.dat', dtype=str)
vol_data = np.genfromtxt('points_vol.dat', skip_header=1)
velocity_data = np.load('data.npy')

labels = np.unique(flow_type[:,1])
label2id = {k:v for k,v in enumerate(labels)}
id2label = {v:k for k,v in label2id.items()}


x_bins = np.linspace(start = vol_data[:,0].min(), stop=vol_data[:,0].max(), num=15)
y_bins = np.linspace(start = vol_data[:,1].min(), stop=vol_data[:,1].max(), num=15)
z_bins = np.linspace(start = vol_data[:,2].min(), stop=vol_data[:,2].max(), num=30)
vol_map_x = np.digitize(vol_data[:,0], x_bins)
vol_map_y = np.digitize(vol_data[:,1], y_bins)
vol_map_z = np.digitize(vol_data[:,2], z_bins)
new_vol_map = np.concatenate((np.expand_dims(vol_map_x, 0), np.expand_dims(vol_map_y, 0), np.expand_dims(vol_map_z, 0)), axis=0).transpose()
velocity_data_sliced = velocity_data[int(flow_type[0][0]):int(flow_type[-1][0])+1, :, :]
new_velocity_data = np.random.rand(10800, 15, 15, 30, 3)
for i in range(len(new_vol_map)):
    pos = new_vol_map[i]
    new_velocity_data[:, pos[0]-1, pos[1]-1, pos[2]-1, :] = velocity_data_sliced[:, i, :]

#%%
velocity_data_sliced = velocity_data[int(flow_type[0,0]):int(flow_type[-1,0])+1, :, :]

print(velocity_data_sliced.shape)
v = velocity_data_sliced[0:6000, :].reshape(6000,19875)
print(v.shape)
targets = np.array([id2label[i] for i in flow_type[0:6000, 1]])
print(targets.shape)

#%%

"""
Three Golden Properties of Object Oriented Programming:
1. Inheritance  -> Subclassing
2. Polymorphism -> Method Overriding
3. Encapsulation-> Subclassing
"""
class ClassifierModel(nn.Module):

    #   Constructor Function:
    def __init__(self, num_classes, id2label, label2id) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding='same'),
            nn.AdaptiveMaxPool3d(output_size=(8,8,16)),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding='same'),
            nn.AdaptiveMaxPool3d(output_size=(4,4,8)),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding='same'),
            nn.AdaptiveMaxPool3d(output_size=(2,2,4)),
            nn.GELU(),
            nn.Dropout(p=0.25),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding='same'),
            nn.AdaptiveAvgPool3d(output_size=(1,1,2)),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=512, out_features=128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=num_classes)
        )
        self.config = {}
        self.config['id2label']=id2label
        self.config['label2id']=label2id
        self.config['num_classes']=num_classes
        self.loss_fc = nn.CrossEntropyLoss(reduction='mean')
        self.model_compiled=False
        self.train_metric = []

    #   Call Function/Default Function:
    def forward(self, input):
        x = input
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x
    
    def predict(self, input):
        """
        If you want to pass one time step: i.e. input.shape = 19875
        input = input.unsqueeze(0)
        input shape has to be (n, 19875)
        """
        output = self(input)
        pred = output.argmax(dim=-1)
        return pred
    
    def predict_classes(self, input):
        pred = self.predict(input)
        return [self.config['id2label'][str(i)] for i in pred]

    def create_optimizer(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad==True]
        self.optimizer = torch.optim.Adam(params=trainable_params, lr = 5e-3)

    def train_one_epoch(self, train_data, test_data, epoch):
        # if self.optimizer is None:
        #     self.create_optimizer()
        """
        The function currently assumes you call this via model.fit()
        model.compile needs to be invoked before model.fit can be used.
        """
        print(f"Currently Training {epoch+1}/{self.max_epochs} epoch")
        for idx, data in enumerate(train_data):
            if idx%10==0:
                print(f"Batch {idx}/{len(train_data)}...")
            one_batch_of_data = data[0]
            one_batch_of_target = data[1]
            self.train()
            self.training_step(one_batch_of_data, one_batch_of_target)

        if self.metric is not None:
            assert test_data is not None, "You must provide test data if you want to print any performance metric"
            assert self.metric in ["accuracy", "f1", "confusion"], "Invalid metric argument"
            self.eval()
            conf_mat = self.confusion_matrix(test_data)
            if self.metric=="confusion":
                print(f"Confusion Matrix after epoch {epoch+1}: \n{conf_mat}")
            elif self.metric=="accuracy":
                acc = self.compute_accuracy(conf_mat)
                print(f"Accuracy after epoch {epoch+1}: {acc*100}%")
                performance_logger = acc
            elif self.metric=="f1":
                p, r, f1 = self.prec_rec(conf_mat)
                print(f"After epoch {epoch+1}:")
                print(f"Precision: {p}\nRecall: {r}\nF1 Scores: {f1}")
                performance_logger = torch.mean(f1)

            self.training_logs(performance_logger, epoch)


    def training_step(self, one_batch_of_data, one_batch_of_target):
        output = self(one_batch_of_data)
        label = one_batch_of_target
        loss = self.loss_fc(output, label)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
    def fit(self, train_data, test_data=None, epochs=1):
        assert self.model_compiled==True, "Model needs to be compiled before the fit function can be called."
        if not os.path.exists("./results/"):
            os.mkdir("./results")
        if not os.path.exists("./results/checkpoints"):
            os.mkdir("./results/checkpoints")
        if not os.path.exists("./results/training_logs"):
            os.mkdir("./results/training_logs")
        self.max_epochs = epochs
        for epoch in range(epochs):
            print("Currently training epoch %s out of %s" %(epoch, epochs))
            self.train_one_epoch(train_data, test_data, epoch)

    def compute_accuracy(self, conf_mat):
        acc = conf_mat.trace()/conf_mat.sum()
        return acc

    def confusion_matrix(self, test_data):
        conf_mat = torch.zeros(size=(self.config['num_classes'], self.config['num_classes']))
        for idx, data in enumerate(test_data):
            input = data[0]
            target = data[1]
            with torch.no_grad():
                output = self(input)
            pred = output.argmax(dim=-1)
            for i in range(len(target)):
                conf_mat[pred[i]][target[i]]+=1  
        return conf_mat
    
    def prec_rec(self, conf_mat):
        p_matrix = torch.zeros(size=len(conf_mat))
        r_matrix = torch.zeros(size=len(conf_mat))
        for i in range(len(conf_mat)):
            p_matrix[i] = conf_mat[i][i]/torch.sum(conf_mat[i])
            r_matrix[i] = conf_mat[i][i]/torch.sum(conf_mat[:,i])
            
        f1_matrix = 2*(p_matrix*r_matrix)/(p_matrix+r_matrix)
        return p_matrix, r_matrix, f1_matrix

    def compile(self, optimizer, loss, metric=None, trainable_params=None):
        """
        Optimizer:  Accepts an optimizer class or a string value that corresponds to a default optimizer class
                        ex.: torch.optim.Adam, "adam", "adagrad" etc.
        Loss:       Accepts an object of a loss class or a string value that corresponds to a default loss class
                        ex.: torch.nn.MSELoss(), "mseloss", "crossentropyloss"
                        custom loss objects can also be passed as long as they are a subclass of _Loss or Module
        Metric:     Accepts a string either "accuracy", "f1", "confusion"
        Trainable:  Accepts a list of parameters which are to be trained
        Params      defaults to all parameters of the model if not specified
        """
        self.model_compiled=True
        if trainable_params is None:
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

        self.metric = metric

    def save_model(self, file_path):
        if not file_path.endswith(".pt"):
            file_path = file_path+".pt"
        torch.save(self, file_path)

    def training_logs(self, eval_metric, epoch):
        self.train_metric.append(eval_metric)
        if self.train_metric[-1]==max(self.train_metric):
            save_path = f"./results/checkpoints/{self.metric}_{self.train_metric[-1]}@epoch_{epoch}.pt"
            self.save_model(save_path)
        with open("./results/training_logs/training_logs.txt", "a") as file:
            file.write(f"{epoch}\t{self.train_metric[-1]}\n")
    
       
#%%
class VortexDataset(Dataset):
    def __init__(self, data, targets) -> None:
        super().__init__()
        self.data = data
        self.labels = targets

    def __getitem__(self, idx):
        data_point = data[idx]
        torch.tensor(data_point)
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.data)

data = torch.tensor(new_velocity_data, dtype = torch.float32)#.reshape(velocity_data_sliced.shape[0],-1)
targets = torch.tensor([id2label[i] for i in flow_type[:, 1]])
dataset = VortexDataset(data, targets)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
num_classes = len(np.unique(targets))
#%%
model = ClassifierModel(num_classes, id2label, label2id)
# print(targets)
# model.fit(data, targets, epochs=5)
# %%