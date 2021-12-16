import json
import pandas as pd
from tqdm import tqdm
import glob
import bz2
from utils import clean, remove_short

files = glob.glob('./Data/twitter-stream/*/*/*/*/*.json.bz2')
print('Numner of files to open:', len(files))
with open('Data/dataset_quote.txt', 'w') as fw:
    for i, file in tqdm(enumerate(files[:])):
        with bz2.open(file, "rb") as f:
            try:
                for line in f.readlines():
                    tw = json.loads(line)
                    # The tweet should be in English, it should be a quote and the quoted text should be in English
                    if 'lang' in tw and \
                        tw['lang'] == 'en' and \
                        'quoted_status' in tw and \
                        tw['quoted_status']['lang'] == 'en': 
                        
                        if 'retweeted_status' in tw:
                            tw = tw['retweeted_status'] # If the tweet is a retweet, take the retweeeted tweet
                            
                        # Collect (full) text and mentions of the quote
                        if 'extended_tweet' in tw:
                            text = tw['extended_tweet']['full_text']
                            mentions = [x['screen_name'] for x in tw['extended_tweet']['entities']['user_mentions']]
                        else:
                            text = tw['text']
                            mentions = [x['screen_name'] for x in tw['entities']['user_mentions']]

                        # Collect (full) text and mentions of quoted tweet
                        if 'extended_tweet' in tw['quoted_status']:
                            qtext = tw['quoted_status']['extended_tweet']['full_text']
                            qmentions = [x['screen_name'] for x in tw['quoted_status']['extended_tweet']['entities']['user_mentions']]
                        else:
                            qtext = tw['quoted_status']['text']
                            qmentions = [x['screen_name'] for x in tw['quoted_status']['entities']['user_mentions']]

                        # Replace newline and tab
                        text = text.replace('\n', ' ').replace(' \t', ' ').replace('\r', ' ').lower()
                        qtext = qtext.replace('\n', ' ').replace(' \t', ' ').replace('\r', ' ').lower()
                        
                        # Standardize mentions
                        for mention in mentions:
                            text = text.replace(mention.lower(), '')
                        for qmention in qmentions:
                            qtext = qtext.replace(qmention.lower(), '')
                        

                        d = tw['id_str'] + '\t' + \
                            tw['quoted_status_id_str'] + '\t' + \
                            text + '\t' + \
                            qtext + '\n'
                        fw.write(d)
            except:
                continue

                
df = pd.read_csv('Data/dataset_quote.txt', sep='\t', engine='python', quoting=3, header=None)
df = df.rename(columns={0:'id1', 1:'id2', 2:'text1', 3:'text2'})

# Clean and remove short tweets, drop duplicates
df['text1_clean'] = df['text1'].apply(clean)
df['text2_clean'] = df['text2'].apply(clean)
df = remove_short(df)
df = df.drop_duplicates(subset='text2_clean', keep='first')
df = df.drop_duplicates(subset='text1_clean', keep='first')
df = df.drop(['text1', 'text2'], axis=1)
df = df.rename(columns={'text1_clean':'text1', 'text2_clean':'text2'})

df[['id1', 'id2', 'text1', 'text2']].to_csv('Data/dataset_quote.csv', sep='\t', index=None)
