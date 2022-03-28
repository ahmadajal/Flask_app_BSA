from flask import Flask, render_template, request, url_for
import os
from joblib import dump, load
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import pandas as pd
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
import spacy

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Create our list of punctuation marks
    punctuations = string.punctuation

    # Create our list of stopwords
    nlp = spacy.load('en_core_web_sm')
    stop_words = spacy.lang.en.stop_words.STOP_WORDS

    # Load English tokenizer, tagger, parser, NER and word vectors
    parser = English()
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

def clean_text(text):
    # Removing spaces and converting text into lowercase
    return text.strip().lower()
#######
app = Flask(__name__)

@app.route('/')
def student():
    return render_template('index.html')

@app.route('/result',methods = ['POST', 'GET'])
def result():
    if request.method == 'POST':
        result = request.form
        print(result.to_dict())
        res = result.to_dict()
        s_in = clean_text(res["review"])
        s_in = " ".join(spacy_tokenizer(s_in))
        df = pd.Series([s_in])
        # load the model
        model = load('text_classifier.joblib')
        pred = model.predict(df)
        print(pred)
        if pred[0] == 1:
            data = {"Sentiment": "positive"}
        else:
            data = {"Sentiment": "Negative"}
        return render_template("result.html", data=data, result=result)
    if request.method == 'GET':
        return "<h3> nothing to show </h3>"

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8080, debug = True)