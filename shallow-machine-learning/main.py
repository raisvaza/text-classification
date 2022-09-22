import pandas as pd
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import naive_bayes, svm
from sklearn.metrics import accuracy_score

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
    return word_tokenize(processed_text)

for i in range(len(train['text_a'])):
    train.loc[i, 'processed_text'] = str(preprocess(train.loc[i, 'text_a']))

for i in range(len(test['text_a'])):
    test.loc[i, 'processed_text'] = str(preprocess(test.loc[i, 'text_a']))

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(train['processed_text'])
Train_X_Tfidf = Tfidf_vect.transform(train['processed_text'])
Test_X_Tfidf = Tfidf_vect.transform(test['processed_text'])

# Naive Bayes
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,train['label'])
predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes Accuracy Score =",accuracy_score(predictions_NB, test['label'])*100, "%")

# SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf,train['label'])
predictions_SVM = SVM.predict(Test_X_Tfidf)
print("SVM Accuracy Score =",accuracy_score(predictions_SVM, test['label'])*100, "%")