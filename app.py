import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn

ps = PorterStemmer()

tfidf = pickle.load(open("resource/vectorizers.pkl", 'rb'))
model = pickle.load(open("resource/model.pkl", 'rb'))

st.title("Email/SMS Spam Classifier")
mssg = st.text_area("Enter Message")


def transform(text):
    text = text.lower()  # to lower the text
    text = nltk.word_tokenize(text)  # to tokenize the text
    y = []
    for i in text:  # to ignore the special charecter
        if i.isalnum():
            y.append(i)

    text = y.copy()  # do not do text=copy because list is mutable do deep copy
    y.clear()

    for i in text:  # to remove stopwords and punctuation
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()

    for i in text:  # to stem each token
        y.append(ps.stem(i))

    return " ".join(y)


if st.button("Predict"):
    # preprocess
    transformed_mssg = transform(mssg)

    # vectorise
    vec_input = tfidf.transform([transformed_mssg])

    # predict
    result = model.predict(vec_input)[0]

    # display
    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")

