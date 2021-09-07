import pandas as pd
import sys
from sentence_transformers import SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, BinaryClassificationEvaluator

# Get the label as the value in the second place of the string (e.g., '(0, 5)' returns 0)
def get_lab(x):
    return int(x[1])
# Binarize the labels as in the original paper: discard 2 and the above values are positive, the values below are negative.
def get_binary_label(x):
    label = int(x)
    if label>=3:
        return 1
    elif label<=1:
        return 0
    else:
        return -1
    
# Normalize labels to [0, 1]
def get_cont_label(x):
    label = float(x)
    return label/6.

model_name = sys.argv[1]
batch_size = 512

# Load train and test sets
df_train = pd.read_csv('paraphrase_dataset_emnlp2017/Twitter_URL_Corpus_train.txt', 
                       sep='\t', header=None)
df_train['label'] = df_train[2].apply(get_lab)
df_train['cont_label'] = df_train['label'].apply(get_cont_label)
df_train['bin_label'] = df_train['label'].apply(get_binary_label)

df_test = pd.read_csv('paraphrase_dataset_emnlp2017/Twitter_URL_Corpus_test.txt', 
                       sep='\t', header=None)
df_test['label'] = df_test[2].apply(get_lab)
df_test['cont_label'] = df_test['label'].apply(get_cont_label)
df_test['bin_label'] = df_test['label'].apply(get_binary_label)

binary_train_examples = [InputExample(texts=[x[0], x[1]], label=x[6]) 
                         for x in df_train[df_train['bin_label']!=-1].values]
cont_train_examples = [InputExample(texts=[x[0], x[1]], label=x[5]) 
                       for x in df_train.values]

binary_test_examples = [InputExample(texts=[x[0], x[1]], label=x[6]) 
                         for x in df_test[df_test['bin_label']!=-1].values]
cont_test_examples = [InputExample(texts=[x[0], x[1]], label=x[5]) 
                       for x in df_test.values]

train_binary_evaluator = BinaryClassificationEvaluator.from_input_examples(binary_train_examples, 
                                                                  batch_size=batch_size, 
                                                                  name='URL-binary_train')
train_cont_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(cont_train_examples, 
                                                                  batch_size=batch_size, 
                                                                  name='URL-cont_train')

test_binary_evaluator = BinaryClassificationEvaluator.from_input_examples(binary_test_examples, 
                                                                  batch_size=batch_size, 
                                                                  name='URL-binary_test')
test_cont_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(cont_test_examples, 
                                                                  batch_size=batch_size, 
                                                                  name='URL-cont_test')

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
test_binary_evaluator(model, output_path=output_path)
test_cont_evaluator(model, output_path=output_path)