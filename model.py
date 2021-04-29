import torch
import torch.nn as nn
import torch.nn.functional as F



class NN(nn.Module):
    def __init__(self, \
        input_dim, \  
        output_dim, \ 
        n_hidden_nonFC = [], \ 
        n_hidden_FC = [10], \  
        dropout_nonFC = 0, \   
        dropout_FC = 0):       
        super(NN, self).__init__()
        self.FC             = True
        self.input_dim      = input_dim       # the size of the input vectors
        self.output_dim     = output_dim      # the size of the output vectors
        self.layers_nonFC   = nn.ModuleList()
        self.layers_FC      = nn.ModuleList()
        self.n_layers_nonFC = len(n_hidden_nonFC)
        self.n_layers_FC    = len(n_hidden_FC)
        self.dropout_nonFC  = dropout_nonFC   # dropout probability for each of the non FC layers (e.g. GRU or LSTM layers)
        self.dropout_FC     = dropout_FC      # dropout probability for each of the FC layers
        self.n_hidden_nonFC = n_hidden_nonFC  # number of nodes in each of the non FC layers (e.g. GRU or LSTM layers)
        self.n_hidden_FC    = n_hidden_FC     # number of nodes in each of the FC layers

        if self.n_layers_nonFC > 0:
            self.FC = False

        # FC layers. They occur after the GRU or LSTM layers (or at the start if there are no other layers)
        
        if self.n_layers_FC > 0:
            if self.n_layers_nonFC==0:
                self.layers_FC.append(nn.Linear(input_dim, n_hidden_FC[0]))
            else:
                self.layers_FC.append(nn.Linear(input_dim*n_hidden_nonFC[-1], n_hidden_FC[0]))
            if self.n_layers_FC > 1:
                for i in range(self.n_layers_FC-1):
                    self.layers_FC.append(nn.Linear(n_hidden_FC[i], n_hidden_FC[(i+1)]))

        # Last layer
        if self.n_layers_FC>0:
            self.last_layer_FC = nn.Linear(n_hidden_FC[-1], output_dim)
        elif self.n_layers_nonFC>0:
            self.last_layer_FC = nn.Linear(input_dim*n_hidden_nonFC[-1], output_dim)
        else:
            self.last_layer_FC = nn.Linear(input_dim, output_dim)

    def forward(self,x):
        if self.FC:
            # Resize from (1,batch_size * input_dim) to (batch_size, input_dim)
            x = x.view(-1,self.input_dim)
        for layer in self.layers_nonFC:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout_nonFC, training=self.training)
            # potentially add the sigmoid for the BCELoss
        if self.n_layers_nonFC > 0:
            x = x.view(-1, self.input_dim*self.n_hidden_nonFC[-1])
        for layer in self.layers_FC:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dropout_FC, training=self.training)
            # potentially add the sigmoid for the BCELoss
        x = self.last_layer_FC(x)
        return x

###############################################################################

class GRU(NN):
    def __init__(self, \
        input_dim, \
        output_dim, \
        n_hidden_nonFC = [2], \
        n_hidden_FC = [], \
        dropout_nonFC = 0.2, \
        dropout_FC = 0):
        super(GRU, self).__init__(\
            input_dim, output_dim, n_hidden_nonFC,\
            n_hidden_FC, dropout_FC, dropout_nonFC)

        self.layers_nonFC.append(nn.GRU(input_dim, n_hidden_nonFC[0]))  # Question: check whether this is correct to do so
        if self.n_layers_nonFC > 1:
            for i in range(self.n_layers_nonFC-1):
                self.layers_nonFC.append(nn.GRU(n_hidden_nonFC[i], n_hidden_nonFC[(i+1)])) # Question: can we stack like that GRUs?

###############################################################################

class LSTM(NN):
    def __init__(self, \
        input_dim, \
        output_dim, \
        n_hidden_nonFC = [2], \
        n_hidden_FC = [], \
        dropout_nonFC = 0.2, \
        dropout_FC = 0):
        super(LSTM, self).__init__(\
            input_dim, output_dim, n_hidden_nonFC,\
            n_hidden_FC, dropout_FC, dropout_nonFC)

        self.layers_nonFC.append(nn.LSTM(input_dim, n_hidden_nonFC[0]))  # Question: check whether this is correct to do so
        if self.n_layers_nonFC > 1:
            for i in range(self.n_layers_nonFC-1):
                self.layers_nonFC.append(nn.GRU(n_hidden_nonFC[i], n_hidden_nonFC[(i+1)])) # Question: can we stack like that GRUs?

###############################################################################

# TODO: RNN

# TODO: CNN

# TODO: TRANSFORMER MODELS -- BERT etc