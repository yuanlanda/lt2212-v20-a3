import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import random
from torch.autograd import Variable
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
import matplotlib.pyplot as plt


ACTIVATION_FUNC = {"relu": nn.ReLU(), "tanh": nn.Tanh()}


class AuthorPredictNN(nn.Module):
  def __init__(self, input_size, hidden_size=0, activation=None):
    super(AuthorPredictNN, self).__init__()

    self.input_size = input_size
    self.hidden_size = hidden_size
    self.activation = activation
    
    if self.hidden_size > 0:
      self.fc1 = nn.Linear(self.input_size * 2, self.hidden_size)
      self.fc2 = nn.Linear(self.hidden_size, 1) 
    else:
      self.fc1 = nn.Linear(self.input_size * 2, 1)
    if self.activation:
      self.nonlinearity = ACTIVATION_FUNC[self.activation]
    
    self.sigmoid = nn.Sigmoid()


  def forward(self, X):
    X = self.fc1(X)
    
    # activation funs for hidden layers
    if self.activation and self.hidden_size > 0:
      X = self.nonlinearity(X)
    if self.hidden_size > 0:
      X = self.fc2(X)
    
    X = self.sigmoid(X)   
    return X


  def train(self, train_X, epochs=20):
    input_size = len(train_X.columns)
    optimizer = torch.optim.Adam(self.parameters(), lr = 0.01)
    criterion = nn.BCELoss()
    train_Y = []
    X1 = []

    for epoch in range(epochs):
      print("epoch:", epoch)
      for author_index,author_docs_indexes in train_dict.items():
        current_author_docs_count = len(author_docs_indexes)
        others_docs_indexes = other_authors_docs_list(author_docs_indexes, len(train_docs_indexes))
        
        for index, doc in enumerate(author_docs_indexes):
          current_author_doc = torch.FloatTensor(np.array(train_X.iloc[doc]))
          random_author_doc = []
          
          if random.randrange(2) == 0:
            random_other_author_doc = random.randrange(len(others_docs_indexes))
            random_author_doc = torch.FloatTensor(np.array(train_X.iloc[others_docs_indexes[random_other_author_doc]]))
            others_docs_indexes.remove(others_docs_indexes[random_other_author_doc])
            train_Y.append(0)
          else:
            # making sure current_author_doc and random_author_doc are not the same docs
            if index < current_author_docs_count - 1:
              random_author_doc = torch.FloatTensor(np.array(train_X.iloc[author_docs_indexes[index + 1]]))
            else:
              continue
            train_Y.append(1)
            
          x = torch.cat((current_author_doc, random_author_doc))
          optimizer.zero_grad()
          outputs = self.forward(x)
          X1.append(outputs)

        X2 = torch.cat(X1, dim = 0)    
        ro = [torch.FloatTensor([pre]) for pre in train_Y]
        ro = torch.FloatTensor(ro)
        X1 = [torch.FloatTensor([pre]) for pre in train_Y]
        loss = criterion(X2, ro)
        loss.backward()
        optimizer.step()


          # data_x = Variable(torch.FloatTensor(x), requires_grad=True)
          # data_y = Variable(torch.FloatTensor([train_Y]))
          # outputs = self.forward(data_x)
          # loss = criterion(outputs, data_y)
          # optimizer.zero_grad()
          # loss.backward()
          # optimizer.step()
      

  def test(self, test_X):
    test_Y = []
    pred = []
    
    with torch.no_grad():
      
      for author_index,author_docs_indexes in test_dict.items():  
        current_author_docs_count = len(author_docs_indexes)
        others_docs_indexes = other_authors_docs_list(author_docs_indexes, len(test_docs_indexes))
        
        for index, doc in enumerate(author_docs_indexes):
            current_author_doc = torch.FloatTensor(np.array(test_X.iloc[doc]))
            random_author_doc = []
            if random.randrange(2) == 0:
              random_other_author_doc = random.randrange(len(others_docs_indexes))
              random_author_doc = torch.FloatTensor(np.array(train_X.iloc[others_docs_indexes[random_other_author_doc]]))
              others_docs_indexes.remove(others_docs_indexes[random_other_author_doc])
              test_Y.append(0)
            else:
              # making sure current_author_doc and random_author_doc are not the same docs
              if index < current_author_docs_count - 1:
                random_author_doc = torch.FloatTensor(np.array(train_X.iloc[author_docs_indexes[index + 1]]))
              else:
                continue
              test_Y.append(1)
            
            x = torch.cat((current_author_doc, random_author_doc))
            outputs = self.forward(x)
            prediction = 1 if outputs > 0.5 else 0
            pred.append(prediction)

    # print(classification_report(test_Y, pred))
    accuracy = accuracy_score(test_Y, pred)
    f = f1_score(test_Y, pred, average='weighted')
    recall = recall_score(test_Y, pred, average='weighted')
    precision = precision_score(test_Y, pred, average='weighted')
    print('accuracy:', accuracy, 'precision:', precision, 'recall:', recall, 'f1_score:', f)
    return(recall, precision)

def create_author_docs_indexes_list(labels):
  labels = labels.to_numpy()
  unique_author_index = np.unique(labels)

  dict = {}
  for i in unique_author_index:
    uni = np.where(labels == i)
    dict[i] = uni[0]

  return labels, dict

# return an index list excluding the author's own files 
def other_authors_docs_list(author_docs, max):
  other_docs_list = [] 
  for i in list(range(max)):
    if i not in author_docs:
      other_docs_list.append(i)
  random.shuffle(other_docs_list)
  
  return other_docs_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test a model on features.")
    parser.add_argument("featurefile", type=str, help="The file containing the table of instances and features.")
    parser.add_argument("--hiddensize", type=int, default=0, dest="hiddensize", help = "The number of hidden layers.")
    parser.add_argument("--activation", type=str, dest="activation", help="Activation function: relu, tanh")
    parser.add_argument("--plotfile", type=str, help = "path of the output plotting image")

    args = parser.parse_args()

    print("Reading {}...".format(args.featurefile))
    
    # read csv file, filter the first index column
    df = pd.read_csv(args.featurefile)
    train_X = df.loc[df['data_type'] == "train"].iloc[:, 1:]
    test_X = df.loc[df['data_type'] == "test"].iloc[:, 1:]

    # separate, generate the files list belongs to each author
    # keys: the indexes of authors, values: a list of all file indexes of the author
    train_docs_indexes, train_dict = create_author_docs_indexes_list(train_X['author'])
    test_docs_indexes, test_dict = create_author_docs_indexes_list(test_X['author'])
    
    train_X = train_X.drop(['data_type','author'], axis = 1)
    test_X = test_X.drop(['data_type','author'], axis = 1)

    if args.plotfile:
      hiddensize = [0, 25, 50, 75, 100]
      for size in hiddensize:
        model = AuthorPredictNN(train_X.shape[1], size)
        model.train(train_X)
        
        recall_list = []
        precision_list = []
        for i in range(5):
          results = model.test(test_X)
          recall_list.append(results[0])
          precision_list.append(results[1])
        
        plt.plot(recall_list, precision_list, label=f"hidden layer size {size}")

      plt.xlabel('Recall')
      plt.ylabel('Precision')

      plt.legend(loc="upper right")
      plt.show()
      plt.savefig(args.plotfile, format='png')

    else:
      model = AuthorPredictNN(train_X.shape[1], args.hiddensize)
      model.train(train_X)
      model.test(test_X)
