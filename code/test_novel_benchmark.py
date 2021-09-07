import pandas as pd
from sentence_transformers.evaluation import RerankingEvaluator_ndcg
from sentence_transformers import SentenceTransformer
import sys

model_name = sys.argv[1] # Name of the model to test
dataset_name = sys.argv[2] # Name of the test dataset: quote, coquote, reply or coreply

df = pd.read_csv('Data/eval_'+dataset_name+'.csv', sep='\t')
df = df.sample(frac=1, random_state=42)

# The first sentence is the anchor, the next 5 are the positive samples, the remaining 25 are negavive samples
def fun(x):
    d = {}
    d['query'] = x[0]
    d['positive'] = x[1:6]
    d['negative'] = x[-26:-1]
    return d

# Take only the first 5000 samples
test_samples = [fun(x) for x in df.values[:5000]]
test_evaluator = RerankingEvaluator_ndcg.RerankingEvaluator_ndcg(test_samples, name=dataset_name+'_ndcg')

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

# Test the model
test_evaluator(model, output_path=output_path)