import pandas as pd
import numpy as np
import torch
import torch.nn as nn

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


class ClassifierModel(nn.Module):
    def __init__(self, num_classes, id2label, label2id) -> None:
        super().__init__()
        self.conv_layer = nn.Conv3d(in_channels=3, out_channels=10, kernel_size=(5,5,5), stride=3, padding='none')
        self.act = nn.GELU()
        self.pool = nn.MaxPool3d(kernel_size=(3,3), stride=1, padding='same')
        self.config = {}
        self.config['id2label']=id2label
        self.config['label2id']=label2id
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.classifier = nn.Linear(in_features=3240, out_features=5)
        self.loss_fc = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = torch.optim.Adam()

    def forward(self, input):
        x = self.conv_layer(input)
        x = self.act(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def create_optimizer(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad=True]
        self.optimizer = torch.optim.Adam(params=trainable_params, lr = 5e-3)

    def train_one_epoch(self, data, target):
        if self.optimizer is None:
            self.create_optimizer()
        for idx in range(len(data)):
            input = data[idx].unsqueeze(0)
            output = self(input)
            label = target[i].unsqueeze(0)
            loss = self.loss_fc(output, label)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        
    def fit(self, data, targets, epochs):
        for epoch in range(epochs):
            self.train_one_epoch(data, targets)

model = ClassifierModel(5, id2label, label2id)
data = new_velocity_data[0:6000]
target = flow_type[0:6000]

model.fit(data, target, 10)
