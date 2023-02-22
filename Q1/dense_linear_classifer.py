import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import TreebankWordTokenizer
from sklearn.metrics import accuracy_score
import nltk


# read the data
df = pd.read_csv("data/labelled_movie_reviews.csv")

# shuffle the rows
df = df.sample(frac=1, random_state=123).reset_index(drop=True)

# get the train, val, test splits
train_frac, val_frac, test_frac = 0.7, 0.1, 0.2
Xr = df["text"].tolist()
Yr = df["label"].tolist()
train_end = int(train_frac*len(Xr))
val_end = int((train_frac + val_frac)*len(Xr))
X_train = Xr[0:train_end]
Y_train = Yr[0:train_end]
X_val = Xr[train_end:val_end]
Y_val = Yr[train_end:val_end]
X_test = Xr[val_end:]
Y_test = Yr[val_end:]

data = dict(np.load("data/word_vectors.npz"))
w2v = {w:v for w, v in zip(data["words"], data["vectors"])}
# convert a document into a vector
tokenizer = TreebankWordTokenizer()
def document_to_vector(doc):
    """Takes a string document and turns it into a vector
    by aggregating its word vectors.

    Args:
        doc (str): The document as a string

    Returns:
        np.array: The word vector this will be 300 dimensionals.
    """
    #  tokenize the input document
    tokens = tokenizer.tokenize(doc)
    text_vecs = [w2v[token] for token in tokens if token in w2v]
    # print(text_vecs)
    # aggregate the vectors of words in the input document
    # calculate mean for all words
    vec = np.mean(text_vecs, axis=0)
    return vec
            

# fit a linear model
def fit_model(Xtr, Ytr, C):
    """Given a training dataset and a regularization parameter
        return a linear model fit to this data.

    Args:
        Xtr (list(str)): The input training examples. Each example is a
            document as a string.
        Ytr (list(str)): The list of class labels, each element of the 
            list is either 'neg' or 'pos'.
        C (float): Regularization parameter C for LogisticRegression

    Returns:
        LogisticRegression: The trained logistic regression model.
    """
    # convert each of the training documents into a vector
    doc_vectors=[document_to_vector(doc) for doc in Xtr]
    # train the logistic regression classifier
    model=LogisticRegression(C=C,solver="sag",max_iter=500)
    model.fit(doc_vectors,Ytr)
    return model

# fit a linear model 
def test_model(model, Xtst, Ytst):
    """Given a model already fit to the data return the accuracy
        on the provided dataset.

    Args:
        model (LogisticRegression): The previously trained model.
        Xtst (list(str)): The input examples. Each example
            is a document as a string.
        Ytst (list(str)): The input class labels, each element
            of the list is either 'neg' or 'pos'.

    Returns:
        float: The accuracy of the model on the data.
    """
    # convert each of the testing documents into a vector
    test_vectors=[document_to_vector(doc) for doc in Xtst]
    # test the logistic regression classifier and calculate the accuracy
    score = model.score(test_vectors,Ytst)
    return score





# search for the best C parameter using the validation set
h_score=-100
h_C = 0
for i in range(-1,11):
    C=2**i
    print(i)
    print('#################')
    model = fit_model(X_train,Y_train,C)
    score = test_model(model,X_val,Y_val)
    if score>h_score:
        h_score=score
        h_C= C
print(h_score)
print(h_C)
# 0.8536
# 64
# fit the model to the concatenated training and validation set
#   test on the test set and print the result
X_new_Train =Xr[0:val_end]
Y_new_Train = Yr[0:val_end]
model = fit_model(X_new_Train,Y_new_Train,64)
score = test_model(model,X_test,Y_test)
print(score)
# 0.8542