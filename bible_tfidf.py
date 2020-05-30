import spacy
from spacy.lang.en import English
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re
import numpy as np

def remove_line_headers(line):
  return re.sub("^\d?[A-Za-z]{1,5}\d*:\d*\w*","", line)

def parse_line_into_components(line):
  dict = {}
  #print(line)
  m = re.search('^(\d?[A-Za-z]{1,5})(\d*):(\d*)\w{1}(.*)$', line)
  dict['book'] = m.group(1)
  dict['chapter'] = m.group(2)
  dict['verse'] = m.group(3)
  dict['text'] = m.group(4)

  return dict


with open('/home/colmnpb/Downloads/kjv/kjv.txt', "r") as f:
  raw_text = f.readlines()

#remove first line (header)
raw_text.pop(0)

text_lines = [remove_line_headers(line) for line in raw_text]

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=500)

x = tfidf_vectorizer.fit_transform(text_lines)

print(x)

print("word tf-idf")
print(tfidf_vectorizer.get_feature_names())
print("\n\n\n\n")

nlp = spacy.load('en_core_web_lg')
print("loaded spacy model")
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
nlp.max_length = 5000000

doc = nlp("".join(text_lines))

word_set = set()

for token in doc:
  word_set.add(token.lemma_)

x = tfidf_vectorizer.fit_transform(list(word_set))

print(x)

indices = np.argsort(tfidf_vectorizer.idf_)[::-1]
features = tfidf_vectorizer.get_feature_names()
print("all lemma tf-idf")
print(features)
print("\n\n\n\n")

print("top features")

#NOTE:  top_features is ordered low to high TF-IDF
top_features = [features[i] for i in indices]

for text in reversed(top_features):
  print(text)