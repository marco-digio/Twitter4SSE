import pandas as pd

ids = set()
def save_ids(name):
    df = pd.read_csv('Data/eval_'+name+'.csv', sep='\t')
    new_ids = set(list(df['id']))
    ids = new_ids | ids
ids = list(ids)

def remove_evals(df):
    for k in list(df.keys()):
        if k[:2] == 'id':
            df = df[~df[k].isin(ids)]
    return df

datasets = ['quote', 'reply', 'coquote', 'coreply']
df_all = pd.DataFrame()
for dataset in datasets:
    df = pd.read_csv('Data/dataset_'+dataset+'.csv', sep='\t')
    df = remove_evals(df)
    df_all = pd.concat([df_all, df.sample(25, replace=False)], axis=0)
    df.to_csv('Data/dataset_'+dataset+'_FINAL.csv', sep='\t')

df_all.sample(frac=1., replace=False).to_csv('Data/dataset_all_FINAL.csv', sep='\t')
