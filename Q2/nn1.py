import torch  
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import json
import gensim.downloader as api
import numpy as np
import torch.nn.functional as F
from nltk.tokenize import TreebankWordTokenizer
import gensim.downloader as api
wv = api.load('glove-wiki-gigaword-200')
print(wv['hello'].shape)
input_d = wv['hello'].shape[0]
# Device configuration 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # select cuda or cpu

# Hyper-parameters 
input_size = 784
hidden_size = 500
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = np.array(train_data['Y'])
# test_data = json.load(open("genre_test.json", "r"))
# Xt = test_data['X']
print(X[0])
print(Y[0])
print(device)

tokenizer = TreebankWordTokenizer()
def sentence_to_vector(doc):
    """Takes a string document and turns it into a vector
    by aggregating its word vectors.

    Args:
        doc (str): The document as a string

    Returns:
        np.array: The word vector this will be 300 dimensionals.
    """
    # TODO: tokenize the input document
    tokens = tokenizer.tokenize(doc)
    text_vecs = [wv[token] for token in tokens if token in wv]
    # print(text_vecs)
    # TODO: aggregate the vectors of words in the input document
    vec = np.mean(text_vecs, axis=0)
    return vec
doc_vectors=np.array([sentence_to_vector(doc) for doc in X])

from torch.utils.data import Dataset, DataLoader
class diabetesdataset(Dataset):
  def __init__(self,x,y):
    self.x = torch.tensor(x,dtype=torch.float32)
    self.y = torch.tensor(y,dtype=torch.float32)
    self.length = self.x.shape[0]
  def __getitem__(self,idx):
    return self.x[idx],self.y[idx]
  def __len__(self):
    return self.length

dataset = diabetesdataset(doc_vectors,Y)
dataloader = DataLoader(dataset=dataset,shuffle=True,batch_size=64)
class NeuralNetwork(torch.nn.Module): 
    def __init__(self, d_in, hid_1,hid_2,hid_3,hid_4,hid_5, d_out):
        super().__init__()
        # TODO: Define layers here
        self.relu = nn.ReLU()
        self.linear1 = torch.nn.Linear(d_in, hid_1)
        self.linear2 = torch.nn.Linear(hid_1, hid_2)
        self.linear3 = torch.nn.Linear(hid_2, hid_3)
        self.linear4 = torch.nn.Linear(hid_3, hid_4)
        self.linear5 = torch.nn.Linear(hid_4, hid_5)
        self.linear6 = torch.nn.Linear(hid_5, d_out)
    def forward(self, x):
        # TODO: Compute output here
        y_1 = self.relu(self.linear1(x))
        y_2 = self.relu(self.linear2(y_1))
        y_3 = self.relu(self.linear3(y_2))
        y_4 = self.relu(self.linear4(y_3))
        y_5 = self.relu(self.linear5(y_4))
        y_6 = self.linear6(y_5)
        return y_6

model = NeuralNetwork(input_d,500,400,200,400,100,4)
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr=0.01)
def train_model(model):
    n_epochs = 30
    for j in range(n_epochs):
      for i,(x_train,y_train) in enumerate(dataloader):
      #prediction
        y_pred = model(x_train)
        y_train = y_train.type(torch.LongTensor)
        cost = criterion(y_pred,y_train)
        #backprop
        # print(cost)
        optimiser.zero_grad()
        cost.backward()
        optimiser.step()
      # print(i)
train_model(model)




# test_data = json.load(open("genre_test.json", "r"))
# Xt = test_data['X']
# model.eval()
# v = torch.Tensor(sentence_to_vector(Xt[10]))
# model(v)
model.eval()
fout = open("ou1.csv", "w")
fout.write("Id,Predicted\n")
index=0
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']
print(len(Xt))
for input in Xt:
  v = torch.Tensor(sentence_to_vector(Xt[10]))

  output = model(v)
  predicted_class_id = int(torch.argmax(output))
  fout.write("%d,%d\n" % (index, predicted_class_id))
  index+=1 