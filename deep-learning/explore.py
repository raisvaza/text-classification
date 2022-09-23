from distutils.command.clean import clean
import enum
import pandas as pd
import re
import nltk
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

## for deep learning
# from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.python.keras import models, layers, backend as K

train = pd.read_csv("data_worthcheck/train.csv")
test = pd.read_csv("data_worthcheck/test.csv")
dev = pd.read_csv("data_worthcheck/dev.csv")

VECTOR_SIZE = 300

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

def tokenize_text(text):
    return nltk.word_tokenize(text)

def remove_stopwords(tokenized_text):
    # token = nltk.word_tokenize(text)
    token_afterremoval = []
    for k in tokenized_text:
        if k not in stopwords_list: 
            token_afterremoval.append(k)

    str_clean = ' '.join(token_afterremoval)
    return token_afterremoval

def preprocess(text):
    processed_text = clean_data(text)
    # processed_text = processed_text.lower()
    processed_text = tokenize_text(processed_text)
    processed_text = remove_stopwords(processed_text)
    return processed_text

# PREPROCESSING: sentence to token

# FEATURE EXTRACTION

# CLASSIFICATION
def find_max_length(data):
    max = len(data[0])
    index = 0
    for i in range(len(data)):
        if max < len(data[i]):
            max = len(data[i])
            
    return max

def pad_sequence(arr, max_length):
    for i in range(len(arr)):
        arr[i] = np.asarray(arr[i])
        for k in range(len(arr[i])):
            arr[i][k] = float(arr[i][k])
        for j in range(len(arr[i]), max_length):
            # arr[i].append(0.0)
            arr[i] = np.append(arr[i], 0.0)
            # np.concatenate(arr[i], np.array([0]))
    return arr

X_dev = dev['text_a'].values
y_dev = dev['label'].values
# print(X_dev)

X_train = train['text_a']
y_train = train['label']

X_test = test['text_a']
y_test = test['label']

# PREPROCESS
for i in range(len(X_dev)):
    X_dev[i] = preprocess(X_dev[i])
# OUTPUT: list of list of token
# print(X_dev)
MAX_COLUMN_LENGTH = find_max_length(X_dev)
# print(MAX_COLUMN_LENGTH)
# print(len(pad_sequence(X_dev, MAX_COLUMN_LENGTH)[0]))
# print(MAX_COLUMN_LENGTH)
# print(len(X_dev[800]))
# print(len(X_dev[0]))

import gensim

# word2vec inputs array of array of token
w2c = gensim.models.word2vec.Word2Vec(sentences=X_dev, min_count=1, vector_size=VECTOR_SIZE)
# print(w2c.wv['agama'].shape)

for i in range(len(X_dev)):
    for j in range(len(X_dev[i])):
        X_dev[i][j] = w2c.wv.key_to_index[X_dev[i][j]]
# print(X_dev)
X_dev = pad_sequence(X_dev, MAX_COLUMN_LENGTH)
# print(X_dev)

"""
for i in range(len(X_train)):
    for j in range(len(X_train[i])):
        X_train[i][j] = w2c.wv.key_to_index[X_train[i][j]]

for i in range(len(X_test)):
    for j in range(len(X_test[i])):
        X_test[i][j] = w2c.wv.key_to_index[test[i][j]]
        """

dic_vocab = w2c.wv.key_to_index

embeddings = np.zeros((len(dic_vocab) + 1, VECTOR_SIZE))
# print(dic_vocab.items())
for word, idx in dic_vocab.items():
    # print(w2c.wv[word])
    try:
        embeddings[idx] =  w2c.wv[word]
    # if word not in model then skip and the row stays all 0s
    except:
        pass

print(type(X_dev))
print(type(X_dev[0]))
print(type(X_dev[0][0]))

x_in = layers.Input(shape=(3,))
# print(len(embeddings[0]))
x = layers.Embedding(input_dim=embeddings.shape[0], 
                    output_dim=embeddings.shape[1], 
                    weights=[embeddings], 
                    input_length=MAX_COLUMN_LENGTH,
                    trainable=False) (x_in)

x = layers.LSTM(units=15, dropout=0.2, return_sequences=True)(x)
                         
x = layers.LSTM(units=15, dropout=0.2)(x)

x = layers.Dense(64, activation='relu')(x)
y_out = layers.Dense(3, activation='softmax')(x)

model = models.Model(x_in, y_out)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

# model.summary()

dic_y_mapping = {n:label for n,label in enumerate(np.unique(y_dev))}
print(dic_y_mapping)
inverse_dic = {v:k for k,v in dic_y_mapping.items()}
print(inverse_dic)
y = np.array([inverse_dic[y] for y in y_dev])
print(y)
X_dev = np.array([[1,2,3],[1,2,3]])
# training = model.fit(x=X_dev, y=y, batch_size=256, 
#                      epochs=10, shuffle=True, verbose=0, 
#                      validation_split=0.3)


print(type(X_dev))
print(type(X_dev[0]))
print(type(X_dev[0][0]))
