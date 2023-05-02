import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


class Net(nn.Module):
    def __init__(self, input, hidden, output, p=0, relu = False, batch = False):
    """
    The layers can be added
     to see the sample, you can check my personal website
     to add some layers just give the list of the number in units
    """
        super(Net, self).__init__()
        self.list_of_layers = []
        self.relu = relu
        self.hidden = hidden
        self.batch = batch
        
        if len(hidden) != 0:
            self.fc1 = nn.Linear(input, hidden[0])
            self.list_of_layers = [nn.Linear(hidden[i],hidden[i+1]) for i in range(len(hidden)-1)]
            self.fc_last = nn.Linear(hidden[-1], output)
        else:
            self.fc1 = nn.Linear(input, output)
            
        self.relu = nn.ReLU()
        self.drpo = nn.Dropout(p)
        self.bn = nn.BatchNorm1d(1000)


    def forward(self, x):
        if len(self.hidden) != 0:
            x = self.fc1(x)
            for layer in self.list_of_layers:
                x = layer(x)
                # if self.batch:
                #     x = self.bn(x)
                if self.relu:
                    x = self.relu(x)
                x = self.drpo(x)

            x = self.fc_last(x)
            return x
        else:
            return self.fc1(x)
            
            
    def train(self, optimizer, loss, lr = 0.03, num_epochs = 10, output_size = 5000, input_size = 10000):
    """
    Adjust all of the hyper parmas here
    """
        
        x = torch.rand(5,input_size)
        labels = torch.rand(1,output_size)
        errs= []
        for epoch in range(num_epochs):
            # Forward pass
            err_of_batch = 0
            for i in range(x.shape[0]):
                features = x[i].reshape(1,x[i].shape[0])
                predict = self.forward(features)
                # L1-Regularization
                err = loss(predict, labels)
                err_of_batch = err_of_batch + err
                # Backward pass and optimization
                optimizer.zero_grad()
                err.backward()
                optimizer.step()
            print("loss "+str (epoch) +" => "+ str(float(err_of_batch)))
            errs.append(np.round(err_of_batch.detach().numpy(),2))
        plt.plot(range(1,num_epochs+1),errs)
        plt.ylabel('Error')
        plt.xlabel('number of epochs')
        plt.show()
        plt.savefig('fig.png')
        return errs, num_epochs


