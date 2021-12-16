import pandas as pd
from tqdm import tqdm
import re
import sys
from collections import Counter
import csv

# eval_name is the name of the direct evaluation dataset: quote or reply
eval_name = sys.argv[1]

def clean(text):
    if type(text)!=str:
        return ''
    text = re.sub(r'http\S+', '', text)
    text = text.replace('@', ' ')
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&gt;', '>')
    text = text.replace('&lt;', '<')
    text = text.lower()
    return " ".join(text.split())

threshold_len = 20
def remove_short(df):
    df['len1'] = df['text1_clean'].apply(len)
    df['len2'] = df['text2_clean'].apply(len)
    df = df[(df['len1']>threshold_len)&(df['len2']>threshold_len)]
    return df

# Select dataset, clean texts, remove short texts
if eval_name == 'quote':
    df = pd.read_csv('Data/dataset_quote.csv', sep='\t', engine='python', quoting=csv.QUOTE_NONE)
    df = df.rename(columns={'id2':'id_star'})
    
elif eval_name == 'reply':
    df = pd.read_csv('Data/dataset_reply.csv', sep='\t', engine='python', quoting=csv.QUOTE_NONE)
    df = df.rename(columns={'id1':'id_star'})
else:
    print('wrong data name!')
    exit()

df['text1_clean'] = df['text1'].apply(clean)
df['text2_clean'] = df['text2'].apply(clean)
df = remove_short(df)
    
# select tweets with more than 4 quotes/replies and remove duplicates
D = Counter(list(df['id_star']))
ids = [k for (k, v) in D.items() if v>4]
df2 = df[df['id_star'].isin(ids)].copy()
if eval_name == 'quote':
    df2 = df2.drop_duplicates(subset='text1_clean', keep='first')
else:
    df2 = df2.drop_duplicates(subset='text2_clean', keep='first')
D2 = Counter(list(df2['id_star']))
ids2 = [k for (k, v) in D2.items() if v>4]

columns_neg = {}
for i in range(25):
    columns_neg[i] = 'neg'+str(i+1)

# For each tweet with more than 4 quotes/replies, sample 5 random quotes/replies and 25 random quotes/replies of other tweets
df_tot = pd.DataFrame()
for id_ in tqdm(ids2[:]):
    df_new_pos = df2[df2['id_star']==id_].sample(5, replace=False)[['text1_clean']].reset_index(drop=True).T
    df_new_pos = df_new_pos.rename(columns={0:'pos1', 1:'pos2', 2:'pos3',3:'pos4',4:'pos5'})
    df_new_neg = df2[df2['id_star']!=id_].sample(25, replace=False)[['text1_clean']].reset_index(drop=True).T
    df_new = pd.concat([df_new_pos, df_new_neg], axis=1)
    df_new['id'] = id_
    df_new['query'] = df2[df2['id_star']==id_].sample(1)['text2_clean'].values[0]
    df_tot = pd.concat([df_tot, df_new])
    
df_tot = df_tot.rename(columns=columns_neg)

names_cols = list(df_tot.columns[:-1].values)
names_cols.insert(0, 'query')

df_tot[names_cols].to_csv('Data/eval_'+eval_name+'.csv', sep='\t', index=None)
