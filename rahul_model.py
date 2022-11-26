#%%
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
#%%
flow_type = np.genfromtxt('../dataset2/FlowStructure_2022_03_24_total.dat', dtype=str)
vol_data = np.genfromtxt('../points_vol.dat', skip_header=1)
velocity_data = np.load('../dataset2/data.npy')

labels = np.unique(flow_type[:,1])
label2id = {k:v for k,v in enumerate(labels)}
id2label = {v:k for k,v in label2id.items()}

"""
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
"""
#%%
velocity_data_sliced = velocity_data[int(flow_type[0,0]):int(flow_type[-1,0])+1, :, :]
#velocity_data_sliced = velocity_data_sliced.astype(np.float32)

#print(velocity_data_sliced.shape)
#v = velocity_data_sliced[0:6000, :].reshape(6000,19875)
#print(v.shape)
#targets = np.array([id2label[i] for i in flow_type[0:6000, 1]])
#print(targets.shape)

#%%

"""
Three Golden Properties of Object Oriented Programming:
1. Inheritance  -> Subclassing
2. Polymorphism -> Method Overriding
3. Abstraction  -> Subclassing
"""
class ClassifierModel(nn.Module):

    #   Constructor Function:
    def __init__(self, num_classes, id2label, label2id) -> None:
        super().__init__()
        #self.conv_layer = nn.Conv3d(in_channels=3, out_channels=10, kernel_size=(5,5,5), stride=3, padding='none')
        self.act = nn.GELU()
        self.mlp1 = nn.Linear(in_features=19875, out_features=9936)
        self.mlp2 = nn.Linear(in_features=9936, out_features=4968)
        self.mlp3 = nn.Linear(in_features=4968, out_features=1000)
        self.mlp4 = nn.Linear(in_features=1000, out_features=200)
        self.mlp5 = nn.Linear(in_features=200, out_features=50)
        self.mlp6 = nn.Linear(in_features=50, out_features=5)
        #self.pool = nn.MaxPool3d(kernel_size=(3,3), stride=1, padding='same')
        self.config = {}
        self.config['id2label']=id2label
        self.config['label2id']=label2id
        #self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        #self.classifier = nn.Linear(in_features=3240, out_features=5)
        self.loss_fc = nn.CrossEntropyLoss(reduction='none')
        # p = self.mlp2.parameters()
        # self.optimizer = torch.optim.Adam(p)

    #   Call Function/Default Function:
    def forward(self, input):
        #x = self.conv_layer(input)
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
        #x = self.pool(x)
        #x = self.flatten(x)
        #x = self.classifier(x)
        return x
    
    def predict(self, input):
        output = self(input)
        #   If you want to pass one time step: i.e. input.shape = 19875
        #   input = input.unsqueeze(0)
        #   input shape has to be (n, 19875)
        pred = output.argmax(dim=-1)
        return pred
    
    def predict_classes(self, input):
        pred = self.predict(input)
        return [self.config['id2label'][str(i)] for i in pred]

    def create_optimizer(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad==True]
        self.optimizer = torch.optim.Adam(params=trainable_params, lr = 5e-3)

    def train_one_epoch(self, data, target, x_test, y_test):
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
            input = data[idx].unsqueeze(0)
            output = self(input)
            label = target[idx].unsqueeze(0)
            loss = self.loss_fc(output, label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def training_step(self, one_batch_of_data, one_batch_of_target):
        output = self(one_batch_of_data)
        label = one_batch_of_target
        loss = self.loss_fc(output, label)
        """...blah blah blah..."""
        
    def fit(self, data, targets, epochs):
        for epoch in range(epochs):
            print("Currently training epoch %s out of %s" %(epoch, epochs))
            self.train_one_epoch(data, targets)

    def compute_accuracy(self, input, target):
        acc = 0
        for idx in range(len(data)):
            input = data[idx].unsqueeze(0)
            output = self(input).squeeze(0)
            pred = output.argmax(dim=-1)
            label = target[idx]
            if pred==label:
                acc+=1
        return float(acc/len(data))

    def confusion_matrix(self, data, target):
        conf_mat = torch.zeros(size=(target.max()+1, target.max()+1))
        for idx in range(len(data)):
            input = data[idx].unsqueeze(0)
            output = self(input).squeeze(0)
            pred = output.argmax(dim=-1)
            label = target[idx]
            conf_mat[label][pred]+=1
        return conf_mat
    
    def prec_rec(self, conf_mat):
        p_matrix = torch.zeros(size=len(conf_mat))
        r_matrix = torch.zeros(size=len(conf_mat))
        for i in range(len(conf_mat)):
            p_matrix[i] = conf_mat[i][i]/torch.sum(conf_mat[i])
            r_matrix[i] = conf_mat[i][i]/torch.sum(conf_mat[:,i])
            
        f1_matrix = (p_matrix*r_matrix)/(p_matrix+r_matrix)
        return p_matrix, r_matrix, f1_matrix

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

        self.metric = metric
                
#%%
data = torch.tensor(velocity_data_sliced[0:6000], dtype = torch.float32).reshape(6000, 19875)
targets = torch.tensor([id2label[i] for i in flow_type[0:6000, 1]])
num_classes = targets.max()+1
# model = ClassifierModel(num_classes, id2label, label2id)
print(targets)
# model.fit(data, targets, epochs=5)