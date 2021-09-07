from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers import evaluation, losses
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import os
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from torch import nn
import pandas as pd
import numpy as np
import gzip
import csv
import sys
import math
import re
from utils import derange

init_model_name = sys.argv[1] # Initialization model: my-vinai/bertweet-base, my-roberta-base, stsb-roberta-base, bert-base-nli-stsb-mean-tokens
n_train = int(sys.argv[2]) # number of training points
dataset_name = sys.argv[3] # Name of the training dataset: quote, reply, coquote, coreply
loss_name = sys.argv[4] # Name of Loss: MultipleNegativesLoss and TripletLoss
batch_size = int(sys.argv[5]) # Batch size 
idx = sys.argv[5] # Index of run (tested on 5 runs)

# if the model name starst with "my-", remove this prefix and use the pretrained model with mean pooling
# otherwise it is a Sentence Model
if init_model_name[:2] == 'my':
    word_embedding_model = models.Transformer(init_model_name[3:], max_seq_length=128)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
else:
    model = SentenceTransformer(init_model_name)

# Datasets allowed
df = pd.read_csv('Data/dataset_'+dataset_name+'_FINAL.csv', sep='\t', index_col='Unnamed: 0')
df = df.sample(frac=1, replace=False, random_state=1)
print('Tot size of dataset: ', df.shape[0])
df_train = df.iloc[:n_train].copy()
print('Training size: ', df_train.shape[0])
    
    
if loss_name == 'MultipleNegativesRankingLoss':
    train_loss = losses.MultipleNegativesRankingLoss(model=model)
    train_samples = [InputExample(texts=[x[2], x[3]]) for x in df_train.values]
elif loss_name == 'TripletLoss':
    train_loss = losses.TripletLoss(model=model)
    df['text3'] = derange(df['text1'].values.copy()) # create negatives
    train_samples = [InputExample(texts=[x[2], x[3], x[4]]) for x in df_train.values]
else:
    print('wrong loss name')
    exit()

train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

# EVALUATE ON STSbenchmark

sts_dataset_path = 'datasets/stsbenchmark.tsv.gz'
if not os.path.exists(sts_dataset_path):
    util.http_get('https://sbert.net/datasets/stsbenchmark.tsv.gz', sts_dataset_path)
    
train_samples = []
dev_samples = []
test_samples = []
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=score)

        if row['split'] == 'dev':
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            test_samples.append(inp_example)
        else:
            train_samples.append(inp_example)
            
evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='sts-dev')


# TRAINING
num_epochs = 1
model_save_path = init_model_name.replace('/', '-')+\
'_'+str(n_train)+'_'+dataset_name+'_'+loss_name+'_'+str(batch_size)+'_'+str(idx)

# Warmup steps 10% of total steps
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)

model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=400,
          warmup_steps=warmup_steps,
          output_path=model_save_path, 
          save_best_model=False)

model.save(model_save_path)


# Load model and test it on STSb test 
model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)
