from tkinter import Image
from flask import Flask, render_template, send_file
import pandas as pd
from requests import Response, head
pd.options.mode.chained_assignment = None
import numpy as np
seed = 0
np.random.seed(seed)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style = 'whitegrid')
from matplotlib.backends.backend_agg import FigureCanvasAgg 
from matplotlib.backends.backend_svg import FigureCanvasSVG 
from matplotlib.figure import Figure
import io
import base64
from PIL import Image

from google_play_scraper import Sort, reviews_all, app
import nltk
import datetime as dt
import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from wordcloud import WordCloud
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report

#getting dataset   
pajak = reviews_all(
      'id.go.pajak.djp',
      sleep_milliseconds=0, 
      lang='id', 
      country='ID', 
      sort = Sort.NEWEST
   )
data_pajak = pd.json_normalize(pajak)
data_pajak_mentah = data_pajak[['content', 'at', 'score']]
# Some functions for preprocessing text
def cleaningText(text):
      text = re.sub(r'#[A-Za-z0-9]+', '', text) # remove hashtag
      text = re.sub(r"http\S+", '', text) # remove link
      text = re.sub(r'[0-9]+', '', text) # remove numbers
      text = re.sub(r'[^\x00-\x7F]+', ' ', text) #remove emoticon
      text = re.sub(r' +', ' ', text) #remove multiple space
      text = text.replace('\n', ' ') # replace new line into space
      text = text.translate(str.maketrans('', '', string.punctuation)) # remove all punctuations
      text = text.strip(' ') # remove characters space from both left and right text
      return text
   #lowercasing
def casefoldingText(text): # Converting all the characters in a text into lower case
      text = text.lower() 
      return text
   #tokenizing
def tokenizingText(text): # Tokenizing or splitting a string, text into a list of tokens
      text = word_tokenize(text) 
      return text
   #filtering stopword
def filteringText(text): # Remove stopwors in a text
      listStopwords = set(stopwords.words('indonesian'))
      filtered = []
      for txt in text:
         if txt not in listStopwords:
               filtered.append(txt)
      text = filtered 
      return text
   #stemming
def stemmingText(text): # Reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words
      factory = StemmerFactory()
      stemmer = factory.create_stemmer()
      text = [stemmer.stem(word) for word in text]
      return text
   #wrapitup
def toSentence(list_words): # Convert list of words into sentence
      sentence = ' '.join(word for word in list_words)
      return sentence
data_pajak_mentah['text_clean'] = data_pajak_mentah['content'].apply(cleaningText) #cleansing
data_pajak_mentah['text_clean'] = data_pajak_mentah['text_clean'].apply(casefoldingText) #casefolding
data_pajak_mentah['text_processing'] = data_pajak_mentah['text_clean'].apply(tokenizingText) #tokenizing
data_pajak_mentah['text_processing'] = data_pajak_mentah['text_processing'].apply(filteringText) #filtering stopword
data_pajak_mentah['text_processing'] = data_pajak_mentah['text_processing'].apply(stemmingText) #stemming
data_pajak_bersih = data_pajak_mentah.copy()
# Determine sentiment polarity of tweets using indonesia sentiment lexicon (source : https://github.com/fajri91/InSet)

# Loads lexicon positive and negative data
data_pajak_bersihv2 = data_pajak_bersih.copy()
lexicon_positive = dict()
import csv
with open('lexicon_positive.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_positive[row[0]] = int(row[1])

lexicon_negative = dict()
import csv
with open('lexicon_negative.csv', 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        lexicon_negative[row[0]] = int(row[1])
        
# Function to determine sentiment polarity of tweets        
def sentiment_analysis_lexicon_indonesia(text):
    #for word in text:
    score = 0
    for word in text:
        if (word in lexicon_positive):
            score = score + lexicon_positive[word]
    for word in text:
        if (word in lexicon_negative):
            score = score + lexicon_negative[word]
    polarity=''
    if (score > 0):
        polarity = 'positive'
    elif (score < 0):
        polarity = 'negative'
    else:
        polarity = 'neutral'
    return score, polarity
#apply lexicon
results = data_pajak_bersihv2['text_processing'].apply(sentiment_analysis_lexicon_indonesia)
results = list(zip(*results))
data_pajak_bersihv2['polarity_score'] = results[0]
data_pajak_bersihv2['polarity'] = results[1]
#show pie chart
fig, ax = plt.subplots(figsize = (6, 6))
sizes = [count for count in data_pajak_bersihv2['polarity'].value_counts()]
labels = list(data_pajak_bersihv2['polarity'].value_counts().index)
explode = (0.1, 0, 0)
ax.pie(x = sizes, labels = labels, autopct = '%1.1f%%', explode = explode, textprops={'fontsize': 14})
ax.set_title('Sentiment Polarity on Reviews M-Pajak', fontsize = 16, pad = 20)
plt.savefig('static/img/pie.png')

#show wordcloud negative
neg_tweets = data_pajak_bersihv2[data_pajak_bersihv2.polarity == 'negative']
neg_string = []
for t in neg_tweets.text_clean:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')
from wordcloud import WordCloud
wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title('Wordcloud Reviews Negative')
plt.savefig('static/img/negword.png')

#show wordcloud positive
neg_tweets = data_pajak_bersihv2[data_pajak_bersihv2.polarity == 'positive']
neg_string = []
for t in neg_tweets.text_clean:
    neg_string.append(t)
neg_string = pd.Series(neg_string).str.cat(sep=' ')
from wordcloud import WordCloud
wordcloud = WordCloud(width=1600, height=800,max_font_size=200).generate(neg_string)
plt.figure(figsize=(12,10))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title('Wordcloud Reviews Positive')
plt.savefig('static/img/posword.png')

#Model LSTM
# Make text preprocessed (tokenized) to untokenized with toSentence Function
X = data_pajak_bersihv2['text_processing'].apply(toSentence) 
max_features = 5000
# Tokenize text with specific maximum number of words to keep, based on word frequency
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(X.values)
X = tokenizer.texts_to_sequences(X.values)
X = pad_sequences(X)
# Encode target data into numerical values
polarity_encode = {'negative' : 0, 'neutral' : 1, 'positive' : 2}
y = data_pajak_bersihv2['polarity'].map(polarity_encode).values
# Split the data (with composition data train 80%, data test 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Create model function with default hyperparameter values
def create_model(embed_dim = 16, hidden_unit = 16, dropout_rate = 0.2, optimizers = Adam, learning_rate = 0.001):
    model = Sequential()
    model.add(Embedding(input_dim = max_features, output_dim = embed_dim, input_length = X_train.shape[1]))
    model.add(LSTM(units = hidden_unit, activation = 'tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(units = 3, activation = 'softmax'))
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizers(lr = learning_rate), metrics = ['accuracy'])
    print(model.summary())
    return model
# From results above, we know the best hyperparameter for this model is :
# {'batch_size': 128, 'dropout_rate': 0.2, 'embed_dim': 64, 'epochs': 10, 'hidden_unit': 64, 'learning_rate': 0.01, 'optimizers': <class 'keras.optimizers.RMSprop'>}
# Create the model with the best hyperparameter which has been determined
model = KerasClassifier(build_fn = create_model,
                        # Model Parameters
                        dropout_rate = 0.2,
                        embed_dim = 64,
                        hidden_unit = 64,
                        optimizers = RMSprop,
                        learning_rate = 0.01,
                   
                        # Fit Parameters
                        epochs=25, 
                        batch_size=128,
                        # Initiate validation data, which is 10% data from data train. It's used for evaluation model
                        validation_split = 0.1)
model_prediction = model.fit(X_train, y_train)
# Visualization model accuracy (train and val accuracy)
fig, ax = plt.subplots(figsize = (10, 4))
ax.plot(model_prediction.history['accuracy'], label = 'train accuracy')
ax.plot(model_prediction.history['val_accuracy'], label = 'val accuracy')
ax.set_title('Model Accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend(loc = 'upper left')
plt.title('Hasil Akurasi Model LSTM')
plt.savefig('static/img/akurasi.png')
#Confusion Matrix
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Akurasi Model LSTM:', accuracy)
print(classification_report(y_test, y_pred))
confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize = (8,6))
sns.heatmap(confusion_matrix(y_true = y_test, y_pred = y_pred), fmt = 'g', annot = True)
ax.xaxis.set_label_position('top')
ax.xaxis.set_ticks_position('top')
ax.set_xlabel('Prediction', fontsize = 14)
ax.set_xticklabels(['negative (0)', 'neutral (1)', 'positive (2)'])
ax.set_ylabel('Actual', fontsize = 14)
ax.set_yticklabels(['negative (0)', 'neutral (1)', 'positive (2)'])
plt.title('Confusion Matrix Model LSTM')
plt.savefig('static/img/confusion.png')

app = Flask(__name__)

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/getdata', methods=('POST', 'GET'))
def getdata():
   ##//##
   return render_template('getdata.html', tables0=[data_pajak.head().to_html()],
               tables2 = [data_pajak_mentah.head().to_html()], titles=[''])

@app.route('/preprocessing', methods=('POST', 'GET'))
def preprocessing():
   return render_template('processing.html', tables1=[data_pajak_bersih.head().to_html()],
                titles=[''])

#result dataframe after lexicon
@app.route('/result', methods=('POST', 'GET'))
def result():
   return render_template('result.html', tables2=[data_pajak_bersihv2.head().to_html()],
               titles=[''])

#about
@app.route('/about', methods=('POST', 'GET'))
def about():
    return render_template('about.html')


if __name__ == '__main__':
   app.run(debug = True)