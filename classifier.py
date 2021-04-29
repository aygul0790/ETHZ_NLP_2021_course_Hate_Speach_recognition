import torch.optim as optim
from torch.optim import lr_scheduler 
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from NLPProject.utils import compute_metrics
from NLPProject.model import *


class Classifier():
    """
        A Neural Network Classifier. A number of Neural Networks (NN) and an MLP are implemented.
        Parameters
        ----------
        input_dim : int
            number of input features.
        output_dim : int
            number of classes.
        n_hidden_nonFC : list, default=[]
            list of integers indicating sizes of other non FC hidden layers (e.g. LSTM or GRU).
        n_hidden_FC : list, default=[]
            list of integers indicating sizes of FC hidden layers. If other type of layers is used, this indicates FC hidden layers after those layers.
        K : integer, default=4
            Convolution layer filter size. Used only when `classifier == 'CNN'`.
        dropout_nonFC : float, default=0
            dropout rate for other hidden layers.
        dropout_FC : float, default=0
            dropout rate for FC hidden layers.
        classifier : str, default='MLP'
            Can be one of the following:
                - 'MLP' : multilayer perceptron
                - 'GRU': GRU Network 
                - 'LSTM': LSTM Network
                - 'RNN': RNN Network 
                - 'CNN': CNN Network
                - 'XXX': XXX Network to be added !!!
                - 'YYY': YYY Network
                - 'ZZZ': ZZZ Network
                - 'Transformer ': Transformer Neural Network to be added !!!
        lr : float, default=0.001
            base learning rate for the SGD optimization algorithm.
        momentum : float, default=0.9
            base momentum for the SGD optimization algorithm.
        log_dir : str, default=None
            path to the log directory. Specifically, used for tensorboard logs.
        device : str, default='gpu'
            the processing unit.



        See also
        --------
        Classifier.fit : fits the classifier to data
        Classifier.eval : evaluates the classifier predictions
    """
    
    def __init__(self,
        input_dim,
        output_dim,
        n_hidden_nonFC=[],
        n_hidden_FC=[],
        K=4,
        dropout_nonFC=0,
        dropout_FC=0, 
        classifier='MLP', 
        lr=.001, 
        momentum=.9,
        log_dir=None,
        device='gpu'):
        if classifier == 'MLP': 
            self.net = NN(input_dim=input_dim, output_dim=output_dim,\
                n_hidden_FC=n_hidden_FC, dropout_FC=dropout_FC)
        if classifier == 'GRU':
            self.net = GRU(input_dim=input_dim, output_dim=output_dim,\
                n_hidden_nonFC=n_hidden_nonFC, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_nonFC=dropout_nonFC)
        if classifier == 'LSTM':
            self.net = LSTM(input_dim=input_dim, output_dim=output_dim,\
                n_hidden_nonFC=n_hidden_nonFC, n_hidden_FC=n_hidden_FC, \
                dropout_FC=dropout_FC, dropout_nonFC=dropout_nonFC)
        # if classifier == 'RNN':
        #     self.net = RNN(input_dim=input_dim, output_dim=output_dim,\
        #         n_hidden_nonFC=n_hidden_nonFC, n_hidden_FC=n_hidden_FC, \
        #         dropout_FC=dropout_FC, dropout_nonFC=dropout_nonFC)
        # if classifier == 'CNN':
        #     self.net = CNN(input_dim=input_dim, output_dim=output_dim,\
        #         n_hidden_nonFC=n_hidden_nonFC, n_hidden_FC=n_hidden_FC, \
        #         dropout_FC=dropout_FC, dropout_nonFC=dropout_nonFC, K=K) 
        # if classifier =="XXX":
        #     self.net = XXX(input_dim=input_dim, output_dim=output_dim,\
        #         n_hidden_nonFC=n_hidden_nonFC, n_hidden_FC=n_hidden_FC, \
        #         dropout_FC=dropout_FC, dropout_nonFC=dropout_nonFC) 
        # if classifier =="YYY":
        #     self.net = YYY(input_dim=input_dim, output_dim=output_dim,\
        #         n_hidden_nonFC=n_hidden_nonFC, n_hidden_FC=n_hidden_FC, \
        #         dropout_FC=dropout_FC, dropout_nonFC=dropout_nonFC) 
        # if classifier =="ZZZ":
        #     self.net = ZZZ(input_dim=input_dim, output_dim=output_dim,\
        #         n_hidden_nonFC=n_hidden_nonFC, n_hidden_FC=n_hidden_FC, \
        #         dropout_FC=dropout_FC, dropout_nonFC=dropout_nonFC) 
        # if classifier =="Transformer":
        #     self.net = Transformer(input_dim=input_dim, output_dim=output_dim,\
        #         n_hidden_nonFC=n_hidden_nonFC, n_hidden_FC=n_hidden_FC, \
        #         dropout_FC=dropout_FC, dropout_nonFC=dropout_nonFC) 
                
        self.criterion = nn.BCELoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=momentum)
        self.logging   = log_dir is not None
        self.device    = device
        if self.logging:
            self.writer = SummaryWriter(log_dir=log_dir,flush_secs=1)
 
    def fit(self,data_loader,epochs,test_dataloader=None,verbose=False):
        
        """
            fits the classifier to the input data.
            Parameters
            ----------
            data_loader : torch dataloader
                the training dataset.
            epochs : int
                number of epochs.
            test_dataloader : torch dataloader, default=None
                the test dataset on which the model is evaluated in each epoch.
            verbose : boolean, default=False
                whether to print out loss during training.
        """    
    
        if self.logging:
            data = next(iter(data_loader))
          #  self.writer.add_graph(self.net,[data.x,XXX]) -- add instead of XXX what we want
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=1, eta_min=0.0005, last_epoch=-1)
        for epoch in range(epochs):
            self.net.train()
            self.net.to(self.device)
            total_loss = 0
            
            for batch in data_loader:
                x, label = batch.x.to(self.device), batch.y.to(self.device) 
                self.optimizer.zero_grad()
                pred  = self.net(x)
                loss  = self.criterion(pred,label)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                total_loss += loss.item() * batch.num_graphs 
            total_loss /= len(data_loader.dataset)
            if verbose and epoch%(epochs//10)==0:
                print('[%d] loss: %.3f' % (epoch + 1,total_loss))

            if self.logging:
                #Save the training loss, the training accuracy and the test accuracy for tensorboard vizualisation
                self.writer.add_scalar("Training Loss",total_loss,epoch)
                accuracy_train = self.eval(data_loader,verbose=False)[0]
                self.writer.add_scalar("Accuracy on Training Dataset",accuracy_train,epoch)
                if test_dataloader is not None:
                    accuracy_test = self.eval(test_dataloader,verbose=False)[0]
                    self.writer.add_scalar("Accuracy on Test Dataset",accuracy_test,epoch)
                           

    def eval(self,data_loader,verbose=False):
        
        """
            evaluates the model based on predictions
            Parameters
            ----------
            test_dataloader : torch dataloader, default=None
                the dataset on which the model is evaluated.
            verbose : boolean, default=False
                whether to print out loss during training.
            Returns
            ----------
            accuracy : accuracy
            conf_mat : confusion matrix
            precision : weighted precision score
            recall : weighted recall score
            f1_score : weighted f1 score
        """  
        
        self.net.eval()

        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in data_loader:
                x, label = batch.x.to(self.device), batch.y.to('gpu')
                y_true.extend(list(label))
                outputs = self.net(x)
                _, predicted = outputs.data
                predicted = predicted.to('gpu')
                y_pred.extend(list(predicted))
                
        accuracy, conf_mat, precision, recall, f1_score = compute_metrics(y_true, y_pred)
        
        if verbose:
            print('Accuracy: {:.3f}'.format(accuracy))
            print('Confusion Matrix:\n', conf_mat)
            print('Precision: {:.3f}'.format(precision))
            print('Recall: {:.3f}'.format(recall))
            print('f1_score: {:.3f}'.format(f1_score))
            
        return accuracy, conf_mat, precision, recall, f1_score