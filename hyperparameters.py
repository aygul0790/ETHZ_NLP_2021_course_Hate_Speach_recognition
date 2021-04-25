from NLPProject.classifier import Classifier

def select_hyperparameters_CV(
    dataset,
    input_dim,
    output_dim,
    n_hidden_nonFC=[10],
    n_hidden_FC=[],
    K=4,             # filter size for CNN
    classifier='MLP', 
    lr=.01, 
    momentum=.9,
    epochs=50,
    device='gpu',
    batch_size=16,
    dropout_rate_list=[0,0.1,0.2,0.5]):
    
    """
    Select the best dropout rate and/or other parameters using cross-validation
    
    TODO: the list of hyper-parameters ot be optimized can be extended!!!
    """

    best_rate=0
    best_score=0
    for dropout_rate in dropout_rate_list:
        
            score=0
            
            for train_dataloader,val_dataloader in dataset.CV_dataloaders(n_splits=4,batch_size=batch_size):
                clf = Classifier(input_dim=input_dim,output_dim=output_dim,classifier=classifier,K=K,n_hidden_FC=n_hidden_FC,n_hidden_nonFC=n_hidden_nonFC,\
                    dropout_nonFC = dropout_rate, dropout_FC=dropout_rate, lr=lr,momentum=momentum,device=device)

                clf.fit(train_dataloader, epochs = epochs, test_dataloader=val_dataloader,verbose=False)
                score+= clf.eval(val_dataloader,verbose=False)[0]

            if score>best_score:
                best_score = score
                best_rate = dropout_rate

    return best_rate



def get_hyperparams(CV_dropout,dataset,input_dim,n_obs_train,output_dim,n_hidden_nonFC,n_hidden_FC,K,classifier,lr,momentum,epochs,\
    device,batch_size,dropout_rate):
    
    """Return the dropout rate, either by doing cross-validation or by using specified values."""
    
    if CV_dropout:
            dropout_rate_list=[0,0.1,0.2,0.5] # grid of dropout rate values
    else:
            dropout_rate_list=[dropout_rate]
                
    dropout_rate = select_hyperparameters_CV(dataset=dataset,input_dim=input_dim,output_dim=output_dim,n_hidden_nonFC=n_hidden_nonFC,n_hidden_FC=n_hidden_FC,\
                K=K,classifier=classifier,lr=0.001,momentum=0.9,epochs=epochs,device=device,batch_size=batch_size,dropout_rate_list=dropout_rate_list)
        
    print("Selected dropout rate: " + str(dropout_rate))
            
    return dropout_rate

