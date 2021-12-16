import json
from tqdm import tqdm
import glob
import bz2
from utils import clean, remove_short
import pandas as pd
import ast

files = glob.glob('./Data/twitter-stream/*/*/*/*/*.json.bz2')
print('Numner of files to open:', len(files))

with open('Data/dataset_reply.txt', 'w') as fw:
    for i, file in tqdm(enumerate(files[:])):
        with bz2.open(file, "rb") as f:
            try:
                for line in f.readlines():
                    tw = json.loads(line)
                    if 'lang' in tw and \
                            tw['lang'] == 'en': # The tweet should be in English

                        if 'retweeted_status' in tw:
                            tw = tw['retweeted_status'] # If the tweet is a retweet, take the retweeeted tweet

                        # Collect (full) text
                        if 'extended_tweet' in tw:
                            text = tw['extended_tweet']['full_text']
                        else:
                            text = tw['text']
                            
                        # Replace newline and tab
                        text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')

                        # Field to identify replies
                        if tw['in_reply_to_status_id_str'] == None:
                            id2 = '0'
                        else:
                            id2 = tw['in_reply_to_status_id_str']
                            
                        # Collect hashtags, mentions and urls
                        hashtags = [x['text'] for x in tw['entities']['hashtags']]
                        mentions = [x['screen_name'] for x in tw['entities']['user_mentions']]
                        urls = [x['url'] for x in tw['entities']['urls']]
                        
                        hashtags = '["'+'", "'.join(hashtags)+'"]'
                        mentions = '["'+'", "'.join(mentions)+'"]'
                        urls = '["'+'", "'.join(urls)+'"]'

                        d = tw['id_str'] + '\t' + id2 + '\t' + tw['user']['id_str'] + '\t ' + text + ' \t'+ hashtags + '\t' + mentions + '\t' + urls + '\n'

                        fw.write(d)
            except:
                continue

df = pd.read_csv('Data/dataset_reply.txt', sep='\t', engine='python', quotechar='‰', header=None)

df = df.rename(columns={0:'id_tw', 
                        1:'id_repl', 
                        2:'user', 
                        3:'text_tw', 
                        4:'hashtags', 
                        5:'mentions',
                        6:'urls'})[['id_tw', 'id_repl', 'text_tw', 'hashtags', 'mentions', 'urls']]

# Remove duplicates
df = df.drop_duplicates(subset='id_tw', keep='first')

# Remove mentions
df['mentions'] = df['mentions'].apply(ast.literal_eval)
def remove_mentions(x):
    text = x['text_tw']
    for m in x['mentions']:
        text = text.replace(m, '')
    return text
df['text_tw'] = df.apply(remove_mentions, axis=1)

# Create dataframe of replied tweets
df_repl = df[(df['id_repl'].isin(df['id_tw'].values))][['id_tw', 'id_repl', 'text_tw']]
#print(df.shape, df[df['id_repl']!=0].shape, df_repl.shape)

# Merge the dataset of replies with the replied tweets
df_ready = df[['id_tw', 'text_tw']].rename(columns={'id_tw':'id_repl', 'text_tw':'text_repl'})
df_merge = df_ready.merge(df_repl, on='id_repl', how='right')
#print(df_merge.shape)

# Clean and remove short tweets, drop duplicates
df2 = df_merge[['id_repl', 'id_tw', 'text_repl', 'text_tw']]
df2['text1_clean'] = df2['text_repl'].apply(clean)
df2['text2_clean'] = df2['text_tw'].apply(clean)
df2 = remove_short(df2)
df2 = df2.drop_duplicates(subset='text2_clean', keep='first')
df2 = df2.drop_duplicates(subset='text1_clean', keep='first')
df2 = df2.rename(columns={'id_repl':'id1', 'id_tw':'id2', 'text1_clean':'text1', 'text2_clean':'text2'})

df2[['id1', 'id2', 'text1', 'text2']].to_csv('Data/dataset_reply.csv', sep='\t', index=None)
