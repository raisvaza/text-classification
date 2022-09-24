import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

import re
import nltk
import pandas as pd
import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize

train = pd.read_csv("data_worthcheck/train.csv")
test = pd.read_csv("data_worthcheck/test.csv")
dev = pd.read_csv("data_worthcheck/dev.csv")

stopwords_list = set(StopWordRemoverFactory().get_stop_words())

# Preprocessing
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
    return word_tokenize(processed_text)

for i in range(len(train['text_a'])):
    train.loc[i, 'token'] = str(preprocess(train.loc[i, 'text_a']))

for i in range(len(test['text_a'])):
    test.loc[i, 'token'] = str(preprocess(test.loc[i, 'text_a']))


# Tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenizing(input_text, tokenizer):
    return tokenizer.encode_plus(   input_text,
                                    add_special_tokens = True,
                                    max_length = 512,
                                    padding = 'max_length',
                                    truncation=True,
                                    return_attention_mask = True,
                                    return_tensors = 'pt'   )

token_id = []
attention_masks = []
labels = []
for sample in train['token']:
    encoding_dict = tokenizing(sample,tokenizer)
    token_id.append(encoding_dict['input_ids']) 
    attention_masks.append(encoding_dict['attention_mask'])

# input_ids = torch.tensor([idx for idx in range(len(train['text_a']))])
token_id = torch.cat(token_id, dim = 0)
attention_masks = torch.cat(attention_masks, dim = 0)
labels = torch.tensor(train['label'].replace(['no','yes'],[0,1]))

# Data loader
dataset = TensorDataset(token_id, attention_masks, labels)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Modelling
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels = 2,
    output_attentions = False,
    output_hidden_states = False,
)

optim = torch.optim.AdamW(  model.parameters(), 
                            lr = 5e-5,
                            eps = 1e-08 )

# uncomment if check gpu True, comment if check gpu False
# model.cuda()

# Training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

epochs = 2

for epoch in range(epochs):
    # setup loop with TQDM and dataloader
    loop = tqdm(loader, leave=True)
    for batch in loop:
        optim.zero_grad()

        batch = tuple(t.to(device) for t in batch)
        token_id, attention_mask, labels = batch
        
        outputs = model(token_id,
                        token_type_ids=None,
                        attention_mask=attention_mask,
                        labels=labels)
        
        loss = outputs.loss
        loss.backward()
        optim.step()

        # print
        loop.set_description(f'Epoch {epoch}')
        loop.set_postfix(loss=loss.item())
