import pandas as pd
import sys
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator

# Get the label as the value in the second place of the string (e.g., '(0, 5)' returns 0)
def get_lab(x):
    return x[1]

# Binarize the labels as in the original paper: on the training set discard 2 and the above values are positive, the values below are negative; in the test set discard 3 and the above values are positive, the values below are negative. The test set is labeled by experts, the train set is not. 
def get_binary_label_train(x):
    label = int(x)
    if label>=3:
        return 1
    elif label<=1:
        return 0
    else:
        return -1
def get_binary_label_test(x):
    label = int(x)
    if label>=4:
        return 1
    elif label<=2:
        return 0
    else:
        return -1
    
# Normalize labels to [0, 1]
def get_cont_label(x):
    label = float(x)
    return label/5.

model_name = sys.argv[1]
batch_size = 512

# Load train, dev and test sets
df_train = pd.read_csv('SemEval-PIT2015-github/data/train.data', 
                 sep='\t', header=None)
df_train['x'] = df_train[4].apply(get_lab)
df_train['cont_label'] = df_train['x'].apply(get_cont_label)
df_train['bin_label'] = df_train['x'].apply(get_binary_label_train)

df_dev = pd.read_csv('SemEval-PIT2015-github/data/dev.data', 
                 sep='\t', header=None)
df_dev['x'] = df_dev[4].apply(get_lab)
df_dev['cont_label'] = df_dev['x'].apply(get_cont_label)
df_dev['bin_label'] = df_dev['x'].apply(get_binary_label_train)

df_test = pd.read_csv('SemEval-PIT2015-github/data/test.data', 
                 sep='\t', header=None)
df_test['cont_label'] = df_test[4].apply(get_cont_label)
df_test['bin_label'] = df_test[4].apply(get_binary_label_test)

binary_train_examples = [InputExample(texts=[x[2], x[3]], label=x[9]) 
                         for x in df_train[df_train['bin_label']!=-1].values]
cont_train_examples = [InputExample(texts=[x[2], x[3]], label=x[8]) 
                       for x in df_train.values]

binary_dev_examples = [InputExample(texts=[x[2], x[3]], label=x[9]) 
                         for x in df_dev[df_dev['bin_label']!=-1].values]
cont_dev_examples = [InputExample(texts=[x[2], x[3]], label=x[8]) 
                       for x in df_dev.values]

binary_test_examples = [InputExample(texts=[x[2], x[3]], label=x[8]) 
                         for x in df_test[df_test['bin_label']!=-1].values]
cont_test_examples = [InputExample(texts=[x[2], x[3]], label=x[7]) 
                       for x in df_test.values]


train_binary_evaluator = BinaryClassificationEvaluator.from_input_examples(binary_train_examples, 
                                                                  batch_size=batch_size, 
                                                                  name='PIT-binary_train')
train_cont_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(cont_train_examples, 
                                                                  batch_size=batch_size, 
                                                                  name='PIT-cont_train')

dev_binary_evaluator = BinaryClassificationEvaluator.from_input_examples(binary_dev_examples, 
                                                                  batch_size=batch_size, 
                                                                  name='PIT-binary_dev')
dev_cont_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(cont_dev_examples, 
                                                                  batch_size=batch_size, 
                                                                  name='PIT-cont_dev')

test_binary_evaluator = BinaryClassificationEvaluator.from_input_examples(binary_test_examples, 
                                                                  batch_size=batch_size, 
                                                                  name='PIT-binary_test')
test_cont_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(cont_test_examples, 
                                                                  batch_size=batch_size, 
                                                                  name='PIT-cont_test')

# Load the model
if model_name[:4] == 'orig': # If it is not pretrained
    try:
        os.mkdir(model_name)
    except OSError:
        print ("Creation of the directory %s failed" % model_name)
    word_embedding_model = models.Transformer(model_name[5:], max_seq_length=512)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
else:
    model = SentenceTransformer(model_name)
    if model_name[:2] != 'my':
        model_name = 'my-'+model_name
output_path = model_name.replace('/', '-')

# Test the model on binarized and continuous data
train_binary_evaluator(model, output_path=output_path)
train_cont_evaluator(model, output_path=output_path)
dev_binary_evaluator(model, output_path=output_path)
dev_cont_evaluator(model, output_path=output_path)
test_binary_evaluator(model, output_path=output_path)
test_cont_evaluator(model, output_path=output_path)