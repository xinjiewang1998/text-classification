from transformers import DistilBertTokenizer
from transformers import TFDistilBertForSequenceClassification
import tensorflow as tf
import pandas as pd
import json
import gc

# load the training data
train_data = json.load(open("genre_train.json", "r"))
X = train_data['X']
Y = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humour, 3 - Crime Fiction
docid = train_data['docid']# these are the ids of the books which each training example came from
print(Y)
# load the test data
# the test data does not have labels, our model needs to generate these
# test_data = json.load(open("genre_test.json", "r"))
# Xt = test_data['X']
# print(len(Xt))

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
train_encodings = tokenizer(X, truncation=True, padding=True)



train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    Y
))
from transformers import TFTrainer, TFTrainingArguments

training_args = TFTrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=16,  # batch size per device during training
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs', 
    eval_steps=100,          
)
num_labels=4
with training_args.strategy.scope():
    trainer_model = TFDistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

trainer = TFTrainer(
    model=trainer_model,                 # Transformers model to be trained
    args=training_args,                  # training arguments
    train_dataset=train_dataset,         # training dataset
)
trainer.train()

fout = open("out.csv", "w")
fout.write("Id,Predicted\n")
index=0
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']
print(len(Xt))
for input in Xt:

  predict_input = tokenizer(input,
                  truncation=True,
                  padding=True, 
                  return_tensors="tf")

  output = trainer.model(predict_input).logits
  predicted_class_id = int(tf.math.argmax(output, axis=-1)[0])
  fout.write("%d,%d\n" % (index, predicted_class_id))
  index+=1 