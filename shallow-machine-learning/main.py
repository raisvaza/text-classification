import pandas as pd
import re
import nltk
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn import svm
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

train = pd.read_csv("../data_worthcheck/train.csv")
test = pd.read_csv("../data_worthcheck/test.csv")
dev = pd.read_csv("../data_worthcheck/dev.csv")

stopwords_list = set(StopWordRemoverFactory().get_stop_words())

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
    train.loc[i, 'processed_text'] = str(preprocess(train.loc[i, 'text_a']))

for i in range(len(test['text_a'])):
    test.loc[i, 'processed_text'] = str(preprocess(test.loc[i, 'text_a']))

Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(train['processed_text'])
Train_X_Tfidf = Tfidf_vect.transform(train['processed_text'])
Test_X_Tfidf = Tfidf_vect.transform(test['processed_text'])

# SVM
SVM1 = svm.SVC(kernel='linear', gamma='auto')
SVM1.fit(Train_X_Tfidf,train['label'])
predictions_SVM = SVM1.predict(Test_X_Tfidf)
print("SVM with linear kernel and auto gamma Accuracy Score =",accuracy_score(predictions_SVM, test['label'])*100, "%")

SVM2 = svm.SVC(kernel='linear', gamma='scale')
SVM2.fit(Train_X_Tfidf,train['label'])
predictions_SVM = SVM2.predict(Test_X_Tfidf)
print("SVM with linear kernel and scale gamma Accuracy Score =",accuracy_score(predictions_SVM, test['label'])*100, "%")

SVM3 = svm.SVC(kernel='rbf', gamma='auto')
SVM3.fit(Train_X_Tfidf,train['label'])
predictions_SVM = SVM3.predict(Test_X_Tfidf)
print("SVM with rbf kernel and auto gamma Accuracy Score =",accuracy_score(predictions_SVM, test['label'])*100, "%")

SVM4 = svm.SVC(kernel='rbf', gamma='scale')
SVM4.fit(Train_X_Tfidf,train['label'])
predictions_SVM = SVM4.predict(Test_X_Tfidf)
print("SVM with rbf kernel and scale gamma Accuracy Score =",accuracy_score(predictions_SVM, test['label'])*100, "%")

SVM5 = svm.SVC(kernel='poly', gamma='auto')
SVM5.fit(Train_X_Tfidf,train['label'])
predictions_SVM = SVM5.predict(Test_X_Tfidf)
print("SVM with poly kernel and auto gamma Accuracy Score =",accuracy_score(predictions_SVM, test['label'])*100, "%")

SVM6 = svm.SVC(kernel='poly', gamma='scale')
SVM6.fit(Train_X_Tfidf,train['label'])
predictions_SVM = SVM6.predict(Test_X_Tfidf)
print("SVM with poly kernel and scale gamma Accuracy Score =",accuracy_score(predictions_SVM, test['label'])*100, "%")

SVM7 = svm.SVC(kernel='sigmoid', gamma='auto')
SVM7.fit(Train_X_Tfidf,train['label'])
predictions_SVM = SVM7.predict(Test_X_Tfidf)
print("SVM with sigmoid kernel and auto gamma Accuracy Score =",accuracy_score(predictions_SVM, test['label'])*100, "%")

SVM8 = svm.SVC(kernel='sigmoid', gamma='scale')
SVM8.fit(Train_X_Tfidf,train['label'])
predictions_SVM = SVM8.predict(Test_X_Tfidf)
print("SVM with sigmoid kernel and scale gamma Accuracy Score =",accuracy_score(predictions_SVM, test['label'])*100, "%")
