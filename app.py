from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import re
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

model = pickle.load(open('model.pkl','rb'))
transformer = pickle.load(open('transformer.pkl','rb'))
app = Flask(__name__)


def removeTag(text):
    cleaned = re.compile(r'<.*?>')
    return re.sub(cleaned," ",text)

def removeSpec(text):
    rem = ' '
    for i in text:
        if i.isalnum():
            rem+=i
        else:
            rem+=" "
    return rem

def removeStopWords(text):
    stopWords = set(stopwords.words('english'))
    words = word_tokenize(text)
    return [w for w in words if w not in stopWords]


def stemming(text):
    ss = SnowballStemmer('english')
    return " ".join([ss.stem(w) for w in text])

def predictor(testNews):
    bow,words = [],word_tokenize(testNews)
    for word in words:
        bow.append(words.count(word))
    word_dict = tfidf.vocabulary_
    inp = []
    for i in word_dict:
        inp.append(testNews.count(i[0]))
    y_pred = GNB.predict(np.array(inp).reshape(1,20000))
    return y_pred

@app.route('/')
def hello():
    return "<h1>Hello</h1>"

@app.route('/predict', methods=['POST'])
def predict():
    testNews = reuest.form['message']
    testNews = removeTag(testNews)
    testNews=removeStopWords(testNews)
    testNews = removeSpec(testNews)
    testNews = stemming(testNews)
    output = prediction[0]
    
    return render_template('index.html', prediction_text = f"{output} ")
if __name__ == '__main__':
    app.run(debug=True, port=8000)