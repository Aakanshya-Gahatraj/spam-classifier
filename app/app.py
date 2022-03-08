import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()

def process_messages(message):
  message= message.lower()  #making msg in lower case
  message= nltk.word_tokenize(message)  #seperating words

  y=[]
  for m in message: #removing special characters
    if m.isalnum():
      y.append(m) 

  message= y[:]
  y.clear()

  for m in message:  #removing stopwords: words that are meaningless
    if m not in  stopwords.words('english') and m not in string.punctuation:
      y.append(m)

  message=y[:]
  y.clear()

  for m in message:
    y.append(ps.stem(m)) #gives root form of words

  return " ".join(y)  

tfidf= pickle.load(open('files/vectorizer.pkl','rb'))
model= pickle.load(open('files/model.pkl','rb'))

st.title("SPAM Classifier")
input_msg= st.text_input("Enter your message")

#preprocess
processed_msg= process_messages(input_msg)

if st.button ('Test'):
  #vectorize
  vector_input= tfidf.transform([processed_msg])

  #predict
  result= model.predict(vector_input)[0]

  if result==1:
      st.header("Spam!")
  else:
      st.header("Not Spam!")