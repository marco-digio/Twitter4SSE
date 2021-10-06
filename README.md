# Twitter4SSE
Official repository of [Exploiting Twitter as Source of Large Corpora of Weakly Similar Pairs for Semantic Sentence Embeddings](https://arxiv.org/abs/2110.02030) (EMNLP2021)

## Citing 

If you find this repository helpful, please cite our publication.

```
@misc{digiovanni2021exploiting,
      title={Exploiting Twitter as Source of Large Corpora of Weakly Similar Pairs for Semantic Sentence Embeddings}, 
      author={Marco Di Giovanni and Marco Brambilla},
      year={2021},
      eprint={2110.02030},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## Instructions

First install the requirements in `requirements.txt`. 

The whole code is extensively based on [SBERT](https://www.sbert.net) . Please take a look also at their [GitHub repository](https://github.com/UKPLab/sentence-transformers) for more details. 

### Download data

**WE DO NOT SHARE THE PROCESSED DATASETS DUE TO PRIVACY POLICIES.**  
However, we share the code to generate it from the raw data. 

First download the data with the `download_and_unzip.sh` script.  
Select the year, month and day you are interested in (we selected November and December 2020).  
The script will download and unzip files from [Twitter Stream Archieve](https://archive.org/details/twitterstream). 

### Generate datasets

First:
Run `generate_quote_dataset.py` to generate the **Quote Dataset (Qt)**.  
Run `generate_reply_dataset.py` to generate the **Reply Dataset (Rp)**.  
Second:  
Run `generate_coquote_dataset.py` to generate the **Co-Quote Dataset (CoQt)**.  
Run `generate_coreply_dataset.py` to generate the **Co-Reply Dataset (CoRp)**.  
The generation of coquote and coreply datasets requires that the quote and reply datasets are already generated.  

The datasets are **not** ready yet since we need to remove tweets in the evaluation datasets, thus we need to generate the evaluation datasets:

Run `generate_direct_evaluation_dataset.py quote` to generate **Direct Quote (DQ)** evaluation benchmark.
Run `generate_direct_evaluation_dataset.py reply` to generate **Direct Reply (DR)** evaluation benchmark.
Run `generate_co_evaluation_dataset.py quote` to generate **Co-Quote (CQ)** evaluation benchmark.
Run `generate_co_evaluation_dataset.py reply` to generate **Co-Reply (CR)** evaluation benchmark.

Finally run `remove_evaluation.py` to generate the 4 FINAL datasets without tweets used on the evaluation benchmarks. 

### Train a model

To train a model, run `train.py model_name N_train training_dataset loss_name batch_size idx` where: 

- `model_name` is the name of the initialization model: we tested `stsb-roberta-base` and `bert-base-nli-stsb-mean-tokens` as sentence models, `roberta_base` and `vinai/bertweet-base` as standard models (thus we need to add a pooling operation by simply prefixing the name with "my-");
- `N_train` is the number of training samples (e.g., 250000);
- `training_dataset` is the name of the training dataset: `quote`, `coquote`, `reply`, `coreply` or `all`;
- `loss_name` is the name of the loss function. We tested `MultipleNegativesRankingLoss` and `TripletLoss`;
- `batch_size` is the size of the batches during the training. We tested values from 2 to 50;
- `idx` is the index of multiple runs. We averaged 5 runs.
 
 
Example to train a model from BERTweet with 250000 samples from quote dataset with MNLoss and batch size 50:
`train.py my-vinai/bertweet-base 250000 quote MultipleNegativesRankingLoss 50 0`  
The model will be saved in a new directory called:
`my-vinai-bertweet-base_250000_quote_MultipleNegativesRankingLoss_50_0`

_NOTE:_ During the training, the final model will be automatically tested on STSb, the results are saved in a file named `similarity_evaluation_sts-test_results.csv`

### Test a model

After training and testing on STSb, we run scripts to test on 4 novel benchmarks and 2 enstablished benchmarks (PIT and TURL). 

#### Novel benchmark

To obtain nDCG as metric, first copy `RerankingEvaluator_ndcg.py` on the `sentence_transformers/evaluation` folder created when you have installed `sentence_transformers` library. 

Run `test_novel_benchmark.py model_name dataset_name` where:

- `model_name` is the name of the directory of the trained model
- `dataset_name` is the name of the test dataset (`quote`, `coquote`, `reply` or `coreply`) 

The results are saved in a file named `RerankingEvaluator_dataset_name_ndcg_results.csv`

Example to test the previously trained model on the quote dataset: 
`test_novel_benchmark.py my-vinai-bertweet-base_250000_quote_MultipleNegativesRankingLoss_50_0 quote`
The results will be saved in `my-vinai-bertweet-base_250000_quote_MultipleNegativesRankingLoss_50_0/RerankingEvaluator_quote_ndcg_results.csv`

To test the model not trained yet (example: the pretrained BERTweet with MEAN of tokens) prefix the model name with `orig-`: 

`test_novel_benchmark.py orig-my-vinai/bertweet-base quote`

If you test an existing model, the results will be saved in a directory prefixing the model with `my-`. E.g., the results of 
`test_novel_benchmark.py stsb-roberta-base quote` 
will be saved in `orig-stsb-roberta-base/RerankingEvaluator_quote_ndcg_results.csv` 

#### PIT benchmark

Download the dataset from `https://github.com/cocoxu/SemEval-PIT2015` 

Run `test_PIT_bechmark.py model_name` where:

- `model_name` is the name of the directory of the trained model

The results are saved in files named 
`similarity_evaluation_PIT-cont_test_results.csv` for the SS task and `binary_classification_evaluation_PIT-binary_test_results.csv` for the PI task. 

The code also test the model on the train and dev datasets. 

Example to test the previously trained model on the quote dataset: 
`test_PIT_bechmark.py my-vinai-bertweet-base_250000_quote_MultipleNegativesRankingLoss_50_0`
The results will be saved in `my-vinai-bertweet-base_250000_quote_MultipleNegativesRankingLoss_50_0/similarity_evaluation_PIT-cont_test_results.csv`

To test the model not trained yet (example: the pretrained BERTweet with MEAN of tokens) prefix the model name with `orig-`: 

`test_PIT_bechmark.py orig-my-vinai/bertweet-base`

If you test an existing model, the results will be saved in a directory prefixing the model with `my-`. E.g., the results of 
`test_PIT_bechmark.py stsb-roberta-base quote` 
will be saved in `orig-stsb-roberta-base/similarity_evaluation_PIT-cont_test_results.csv`

#### TURL benchmark

Download the dataset from `https://languagenet.github.io/`

Run `test_TURL_bechmark.py model_name` where:

- `model_name` is the name of the directory of the trained model

The results are saved in files named 
`similarity_evaluation_TURL-cont_test_results.csv` for the SS task and `binary_classification_evaluation_TURL-binary_test_results.csv` for the PI task. 

The code also test the model on the train dataset. 

Example to test the previously trained model on the quote dataset: 
`test_TURL_bechmark.py my-vinai-bertweet-base_250000_quote_MultipleNegativesRankingLoss_50_0`
The results will be saved in `my-vinai-bertweet-base_250000_quote_MultipleNegativesRankingLoss_50_0/similarity_evaluation_TURL-cont_test_results.csv`

To test the model not trained yet (example: the pretrained BERTweet with MEAN of tokens) prefix the model name with `orig-`: 

`test_TURL_bechmark.py orig-my-vinai/bertweet-base`

If you test an existing model, the results will be saved in a directory prefixing the model with `my-`. E.g., the results of 
`test_TURL_bechmark.py stsb-roberta-base quote` 
will be saved in `orig-stsb-roberta-base/similarity_evaluation_TURL-cont_test_results.csv`

#### STSbenchmark

The model is tested on STSb automatically during training. 


