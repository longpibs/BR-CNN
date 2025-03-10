# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 14:05:04 2023

@author: longp

This is for the test of training binary class while validating the results againist all the classes using the binary trained network from a specific class
To evaluate the similarity between class score which could be cauzed by pure incidence or some kind of hidden similarity of classes
"""

import pandas as pd
import pickle
import torch
import h5py
import torch.nn as nn
import csv
import torch.nn.functional as F
from torch.autograd import Variable
import argparse
import torch.optim as optim
import numpy as np
import time
import torch.utils.data as data_utils
from scipy.stats import sem
from sklearn.metrics import auc,f1_score,precision_score,recall_score,hamming_loss,roc_auc_score,precision_recall_fscore_support,accuracy_score,confusion_matrix,roc_curve
from sklearn.model_selection import train_test_split

np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)

def createLossAndOptimizer(net,arg,weight_scale):
    """
    Function for creating loss function and optimizer for the network
    param:
        net: the object of the network
        weight_scale: the weight_scale used in training network to help counter the uneven pos and neg sample problem
        arg: list of hyperparameters
    """
    #Loss function
    loss = torch.nn.CrossEntropyLoss(weight = weight_scale)
    
    #Optimizer
    if arg.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=arg.lr)
    elif arg.optimizer == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters(), rho = 0.95, lr=arg.lr)
    return(loss, optimizer)
    
        
class CNN_Text(nn.Module):
    """
    Class for building WS-CNN using pytorch package
    """
    def __init__(self, args, w2v):
        super(CNN_Text, self).__init__()
        self.args = args
        
        
        ## Defining the parameters
        V =  args.embed_num            # Total number of word vectors
        D =  args.embed_dim            # The length of each word vector
        C =  args.class_num            # Total number of categories for evaluating
        Ci = 1             
        Co = args.kernel_num           # The number of kernels to use in the convolutional layer
        Ks = args.kernel_sizes         # The kernel sizes for W-CNN
        
        ## Defining the layers
        self.embed = nn.Embedding(V, D)                                           # Word embedding layer for W-CNN
        self.embed.weight.data.copy_(torch.from_numpy(w2v))
        self.conv_word = nn.ModuleList([nn.Conv2d(Ci, Co, (K,D)) for K in Ks])    # defining list of convolutional layers for W-CNN

#        for item in self.conv_word:
#            item.weight.data.uniform_(-0.01,0.01)
#            item.bias.data.fill_(0)

        self.dropout = nn.Dropout(args.dropout)
#       self.fcfullconv = nn.Linear(Co,200)
        self.fcfinal = nn.Linear(len(Ks)*Co, C)                  # Last layer that concatenate all feature maps
        

    def forward(self, x_word):
        """
        Defining details of forward pass for WS-CNN
        variables:
            x1: input that go through W-CNN
            x2：input that go through S-CNN
            x : concatenated features from W-CNN and S-CNN
        """
        x1 = self.embed(x_word)
        
        ## summ pooling
#        x2 = x2.sum(-2)
        
        
        ## workflow for word CNN
        x1 = x1.unsqueeze(1)  
        x1 = [F.relu(conv(x1)).squeeze(3) for conv in self.conv_word]    # convolution layer
        x1 = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x1]         # pooling layer
        #x = [F.elu(conv(x)).squeeze(3) for conv in self.convs1]         # try elu if you find problems of dead units
        
        

        x1 = torch.cat(x1, 1)
    
        
        x = self.dropout(x1)  # (N, len(Ks)*Co)
        logit = self.fcfinal(x)  # (N, C)
        return logit
    
    def resetzeropadding(self):
        """
        Resetting 'padding' word vectors to all 0 so padding won't affect the performance
        """
        parameters = self.embed.state_dict()
        parameters['weight'][0] = 0
        self.embed.load_state_dict(parameters)
        

    def l2norm(self,args):
        """
        L2 normalization applied to the last layer of the network ('x' in the forward function this case)
        """
        wei = self.fcfinal.state_dict()
        for j in range(0, wei['weight'].size(0)):
            normnum = wei['weight'][j].norm()
            wei['weight'][j].mul(args.l2s).div(1e-7 + normnum)
        self.fcfinal.load_state_dict(wei)
    

def trainNet(args,net,train_loader,val_loader,weight_scale):
    """
    Function for defining the details on the procedure of training the network
    param:
        args: list of parameters
        net: WS-CNN object
        train_loader: pytorch dataloader that contains training data
        weight_scale: pre-defined scale of weights for balancing the uneven distribution of pos and neg samples in calculating the loss.
        phenotypedictinverse: dictionary that map the predict label back to phenotype name
    """
    multiresults = list()
    best_f1 = 0
    loss, optimizer = createLossAndOptimizer(net,args,weight_scale)
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", args.batch_size)
    print("epochs=", args.epochs)
    print("Optimizer= ",args.optimizer)
    print("learning_rate=", args.lr)
    print("predict_class=", args.index_test_value)
    print("=" * 30)
    training_start_time = time.time()
    n_batches = len(train_loader)
    if args.cuda > -1:
        net.cuda()
    for epoch in range(args.epochs):
        
        ## if we are using gpu, store the network in gpu memory
            
        net.train() # train mode activates the dropout and batchnorm property of network
        running_loss = 0.0
        print_every = n_batches // 10  # Print loss for every  n_batches // 10 iterations
        start_time = time.time()
        
        for i, data in enumerate(train_loader,0):
            
            inputs,labels = data  # fetch data from loader
                
            if args.cuda > -1:                      # transform data into GPU memory for GPU calculating if available
                inputs,labels = inputs.cuda(),labels.cuda() 
            
            
            optimizer.zero_grad()                   # Set the collected gradients to zero
            outputs = net(inputs) # Activate the forward function of the network
            loss_size = loss(outputs, labels)       # calculate the loss according to the loss function
            loss_size.backward()                    # pass the gradients back through the network
            optimizer.step()                        # Updating the weights using the optimizer 
            

            net.resetzeropadding()                  # reseting padding word vector to 0
            net.l2norm(args)                        # l2 normalization to the last layer
            
            ## Print loss every 10th batch of an epoch
            running_loss += loss_size.item()
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                        epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))
                #Reset running loss and time
                running_loss = 0.0
                start_time = time.time()
#        net.printembweight()
        ## Validation for debugging and tuning
        if (epoch % 1) == 0:
            net.eval()
            with torch.no_grad(): 
                total_val_loss = 0
                val_pred_labels = torch.FloatTensor([])
                val_true_labels = torch.FloatTensor([])
                for i, data in enumerate(val_loader,0):
                    inputs,labels = data
                    if args.cuda > -1:
                        inputs,labels = inputs.cuda(),labels.cuda() 
                
                    val_outputs = net(inputs)
                    val_loss_size = loss(val_outputs, labels)
                    total_val_loss += val_loss_size.item()
                    
                    _, val_out_label = torch.max(val_outputs, 1)
                    val_pred_labels = torch.cat((val_pred_labels,val_out_label.type(torch.FloatTensor)))
                    val_true_labels = torch.cat((val_true_labels,labels.type(torch.FloatTensor)))
                    
                val_pred_labels = val_pred_labels.cpu().numpy()
                val_true_labels = val_true_labels.cpu().numpy()
                print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
                scores_val = precision_recall_fscore_support(val_true_labels,val_pred_labels,pos_label = 1, average = 'binary')
                print("Validatiob PRECISION = {:.4f}, RECALL =  {:.4f}, F1 = {:.4f}".format(scores_val[0],scores_val[1],scores_val[2]))
    #            val_true_label_num = torch.argmax(val_true_label_num, dim=1)
    
                multiresults.append([scores_val[0],
                                    scores_val[1],
                                    scores_val[2],
                                    val_pred_labels,
                                    val_true_labels,
                                    np.sum(val_true_labels)/len(val_true_labels),
                                    args.index_test_value,
                                    args.predict_class,
                                    epoch,
                                    time.time() - training_start_time])
                if scores_val[2] >= best_f1:
                    best_f1 = scores_val[2]
                    print("Saving best model to " + args.save_model)
                    torch.save(net.state_dict(),args.save_model + 'model_class_' + str(args.predict_class))
       # net.printembweight()
       
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return multiresults,loss

def readh5todata(path,pathcsv):
    """
    Reading the dataset from the HDF5 file
    param:
        args: list of parameters
        path: the disk location of the dataset
    """
    myFile = h5py.File(path, 'r')
    
    ## read the data and put them in Tensor format

    x_train = torch.LongTensor(myFile['train'][...])
    y_train = torch.LongTensor(myFile['train_label'][...])
    x_test = torch.LongTensor(myFile['test'][...])
    y_test = torch.LongTensor(myFile['test_label'][...])
    
    w2v = myFile['w2v'][...]                                                   # word embedding matrixs
    
#    test = y_train.numpy()
#    test2 = np.sum(test,axis = 0)
#    df_test = pd.DataFrame(test2,columns = ['freq'])
#    df_test['perc'] = df_test['freq'] / 53840
#    df_test = df_test.sort_values(by = 'perc')
#    df_test = df_test.reset_index()
    
    
    df_class_rev  = pd.read_csv(pathcsv)  
    dict_class_rev = df_class_rev.set_index('0').T.to_dict('list')
    dict_class_rev = {v[0]: k for k, v in dict_class_rev.items()}
    
    
    
    
    
    return x_train,x_test,y_train,y_test,w2v,dict_class_rev



#0,1,29,30,31,32,40,53
all_df_multiresults = list()

x_train,x_test,y_train,y_test,emb,dict_class_rev = readh5todata('data_AAPD.h5','data_AAPD_dict.csv')


for i_class in range(y_train.size()[1]):
    

    ## Defining a parser for storing parameters and further propose
    parser = argparse.ArgumentParser(description='CNN text classificer')
    parser.add_argument('-index_test_value', type=int, default=i_class, help='test label')
    parser.add_argument('-predict_class', type=str, default= dict_class_rev[i_class], help= 'Choose which type of phenotyping to detect') 
    parser.add_argument('-lr', type=float, default=0.00025, help='initial learning rate [default: 0.5]')
    parser.add_argument('-l2s', type=float, default=3, help='l2 norm')
    parser.add_argument('-epochs', type=int, default=20, help='number of epochs for train [default: 20]')
    parser.add_argument('-batch_size', type=int, default=128, help='batch size for training [default: 64]')
    #{"Cancer":11,"Heart":4,"Lung":5,"Neuro":10,"Pain":9,"Alcohol":7,"Substance":8,"Obesity":1,"Disorders":6,"Depression":12})
    parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-optimizer', type=str, default='Adam', help='optimizer for the gradient descent: Adadelta, Adam')
    parser.add_argument('-embed-dim', type=int, default=emb.shape[1], help='number of embedding dimensio n [default: 50]')
    parser.add_argument('-embed-num', type=int, default=emb.shape[0], help='number of embedding numbers [default: 48849]')
    parser.add_argument('-class_num', type=int, default=2, help='number of catagories [default: 2]')
    parser.add_argument('-kernel-num', type=int, default=400, help='number of each kind of kernel [default: 100]')
    parser.add_argument('-kernel-sizes', type=str, default='1,2,3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-cuda', type=int, default= 0, help='-1 for cpu, 0 for gpu, multigpu please specify the number of gpu')
    parser.add_argument('-save-model', type=str, default= 'AAPDBRModel\\', help='save best performing model to dir')
    args = parser.parse_args()
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    
    weight_scale = [ 1/ (y_train[:,i_class].size()[0] - np.sum(y_train[:,i_class].numpy().copy()))  , 1/(np.sum(y_train[:,i_class].numpy().copy()))]

    ## if we are using gpu (much faster performance)
    if args.cuda > -1:
        weight_scale = torch.FloatTensor(weight_scale).cuda()
        torch.backends.cudnn.benchmark = False
        torch.cuda.set_device(args.cuda)
        
    
    
    
    
    trainx = data_utils.TensorDataset(x_train, y_train[:,i_class])  # pack all the training data into one Tensor dataset object (since we are using pytorch)
    testx = data_utils.TensorDataset(x_test,y_test[:,i_class])
    train_loader = torch.utils.data.DataLoader(trainx, batch_size=args.batch_size, sampler= None, shuffle =  False)
    test_loader = torch.utils.data.DataLoader(testx, batch_size= args.batch_size, sampler= None, shuffle =  False)
    w2v = emb
    
    
    CNN = CNN_Text(args,w2v) # create the network object
    print(CNN)
    multiresults,loss = trainNet(args,CNN,train_loader,test_loader,weight_scale) # start training the network
    
    df_multiresults = pd.DataFrame(data = multiresults, columns=['precision','recall','f1','pred','true','freq','class_index','class','epoch','time'])
    df_multiresults = df_multiresults.sort_values(by = ['f1'],ascending= False)
    all_df_multiresults.append(df_multiresults.iloc[0])
    
    
    del trainx,testx
    del train_loader,test_loader,w2v
    del CNN,loss
    del weight_scale
    del multiresults,df_multiresults
    time.sleep(5)
    torch.cuda.empty_cache()
    time.sleep(5)
    





results = pd.concat(all_df_multiresults)
#results = new_result
#test_f1_val = results[['f1','f1_val']]
pred_list = list(results['pred'].values)
pred = np.zeros([len(pred_list[0]),len(pred_list)])
true_list = list(results['true'].values)
true = np.zeros([len(true_list[0]),len(true_list)])


for i in range(len(pred_list)):
    pred[:,i] = pred_list[i]
    true[:,i] = true_list[i]


micro = f1_score(y_true=true, y_pred=pred, average='micro')
macro = f1_score(y_true=true, y_pred=pred, average='macro')

print("=============================================================")
print("Training finished, all the best performing models are saved to path: " + args.save_model)
print("MicroF1 : " + str(micro))
print("MacroF1 : " + str(macro))