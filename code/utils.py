import re
import random

def swap(xs, a, b):
    xs[a], xs[b] = xs[b], xs[a]

def permute(xs):
    for a in range(len(xs)):
        b = random.choice(xrange(a, len(xs)))
        swap(xs, a, b)
        
def derange(xs):
    for a in range(1, len(xs)):
        b = random.choice(range(0, a))
        swap(xs, a, b)
    return xs


# Clean function for reply and quote
def clean(text):
    if type(text)!=str:
        return ''
    #text = demoji.replace(text, ' ')
    text = re.sub(r'http\S+', '', text)
    text = text.replace('@', ' ')
    #text = text.replace('#', ' ')
    text = text.replace('\n', ' ').replace('\t', ' ').replace('\r', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&gt;', '>')
    text = text.replace('&lt;', '<')

    text = text.lower()
    return " ".join(text.split())

# Remove short texts function for reply and quote
def remove_short(df):
    df['len1'] = df['text1_clean'].apply(len)
    df['len2'] = df['text2_clean'].apply(len)
    threshold_len = 20
    df = df[(df['len1']>threshold_len)&(df['len2']>threshold_len)]
    return df