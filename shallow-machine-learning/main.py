import pandas as pd
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

train = pd.read_csv("data_worthcheck/train.csv")
test = pd.read_csv("data_worthcheck/test.csv")
dev = pd.read_csv("data_worthcheck/dev.csv")

def clean_data(text):
    normal_tw = text
    normal_tw = text.lower()
    normal_tw = re.sub(r'\\x.{2}', '', normal_tw)
    normal_tw = re.sub(r'((www\.[^\s]*)|(https?://[^\s]*))', '', normal_tw)
    normal_tw = normal_tw.strip()
    normal_tw = re.sub(r'@[^\s]+', '', normal_tw)
    normal_tw = re.sub(r'#[^\s]+', '', normal_tw)
    normal_tw = re.sub(r'\d+', ' ', normal_tw) 
    normal_tw = re.sub(r'^nan$', '', normal_tw) 
    normal_tw = re.sub(r'[_]+', '', normal_tw)
    normal_tw =  re.sub(r'[Ã°Âã¯¹¢²ðƒâ]', '', normal_tw) 
    normal_regex = re.compile(r"(.)\1{1,}")
    normal_tw = normal_regex.sub(r"\1\1", normal_tw)
    normal_tw = re.sub(r'\s+', ' ', normal_tw)
    normal_tw = re.sub(r'[^\w\s]', '', normal_tw) 
    normal_tw = re.sub(r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]', '', normal_tw)
    
    return normal_tw

stopwords_list = set(StopWordRemoverFactory().get_stop_words())

def remove_stopwords(text):
    token = nltk.word_tokenize(text)
    token_afterremoval = []
    for k in token:
        if k not in stopwords_list: 
            token_afterremoval.append(k)

    str_clean = ' '.join(token_afterremoval)
    return str_clean

def preprocess(text):
    processed_text = clean_data(text)
    processed_text = processed_text.lower()
    processed_text = remove_stopwords(processed_text)
    return processed_text
