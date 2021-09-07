import pandas as pd
from tqdm import tqdm
import re
from collections import Counter
import csv

df = pd.read_csv('Data/dataset_quote.csv', sep='\t', 
                 engine='python', quoting=csv.QUOTE_NONE)

# Select tweets with more than one quote and drop duplicates
D = Counter(list(df['id2']))
ids = [k for (k, v) in D.items() if v>1]
df2 = df[df['id2'].isin(ids)].copy()
df2 = df2.drop_duplicates(subset='text1', keep='first')
D2 = Counter(list(df2['id2']))
ids2 = [k for (k, v) in D2.items() if v>1]

# For each tweet with more than 1 quote, sample 2 random quotes
df_tot = pd.DataFrame()
for id_ in tqdm(ids2[:]):
    df_new = pd.DataFrame(df2[df2['id2']==id_].sample(2, replace=False)[['id1', 'text1']].values.flatten()).T
    df_new['id'] = id_
    df_tot = pd.concat([df_tot, df_new])
    
df_tot = df_tot.rename(columns={0:'id1', 1:'text1', 2:'id2', 3:'text2'})
df_tot[['id1', 'id2', 'text1', 'text2', 'id']].to_csv('Data/dataset_coquote.csv', sep='\t', index=None)

    
    
