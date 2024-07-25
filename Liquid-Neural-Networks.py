import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class LiquidNeuralNetwork(nn.Module):
    def __init__(self,input_size,hidden_size,num_layers):
        super(LiquidNeuralNetwork,self).__init__()
        self.hidden_size=hidden_size
        #self.input_size=input_size
        self.num_layers=num_layers
        self.layers=nn.ModuleList([self._create_layer(input_size,hidden_size) for _ in range(num_layers)])

    def _create_layer(self,input_size,hidden_size):
        return nn.Sequential(
            nn.Linear(input_size,hidden_size),
            nn.LeakyRelu(),
            nn.Linear(hidden_size,hidden_size)
        )
    def forward(self,x):
        for i ,layer in enumerate(self.layers):
            x=layer(x)
        return x
class ODESolver(nn.Module):
    def __init__(self,model,dt):
        super(ODESolver,self).__init__()
        self.model=model
        self.dt=dt
    def forward(self,x):
        with torch.enable_grad():
            outputs=[]
            for i,layer in enumerate(self.model):
                outputs.append(layer(x))
                x=outputs[-1]
        return x
    def loss(self,x,t):
        with torch.enable_grad():
            outputs=[]
            for i,layer in enumerate(self.model):
                outputs.append(layer(x))
                x=outputs[-1]
        return x
def train(model,dataset,optimizer,epochs,batch_size):
    model.train()
    total_loss=0
    for epoch in range(epochs):
        for batch in dataset:
            inputs,labels=batch
            optimizer.zero_grad()
            outputs=model(inputs)
            loss=model.loss(inputs,outputs)
            loss.backward()
            optimizer.step()
            total_loss+=loss.item()
        print(f'Epoch {epoch+1},loss: {total_loss/len(dataset)}')
