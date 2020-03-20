import spacy
from spacy.lang.en import English
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import re

def remove_line_headers(line):
    return re.sub("^\d?[A-Z][a-z]{1,2}\d*:\d*\w*","", line)

with open('/home/colmnpb/Downloads/kjv/kjv.txt', "r") as f:
  raw_text = f.readlines()


raw_lines = [remove_line_headers(line) for line in raw_text ]

#for line in raw_lines:
#    print(line)



#nlp = English();
nlp = spacy.load('en_core_web_lg')
print("loaded spacy model")
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
nlp.max_length = 5000000 #to handle entire text of bible
#sbd = nlp.create_pipe('sentencizer')
#nlp.add_pipe(sbd)


doc = nlp("".join(raw_lines))
#doc = nlp("".join(raw_lines[1:100]))

verbs = {}
for token in doc:

  #print(token, "::", token.pos_)

  if token.pos_ == "VERB":
    print(token, "-->", token.lemma_)
    if not token.lemma_ in verbs:
      verbs[token.lemma_] = 1
    else:
      verbs[token.lemma_] = verbs[token.lemma_] + 1


# with open('verbs.csv', "w") as f:
#   for key in verbs:
#    f.write(key)
#    f.write(", ")
#    f.write( str(verbs[key]))
#    f.write("\n")

with open('raw_adj.txt', "r") as f:
  raw_adjs = f.readlines

adj_doc = nlp("".joine(raw_adjs))



exit()

sents_list = []
for sent in doc.sents:
    sents_list.append(sent.text)
#print(sents_list)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)

x = tfidf_vectorizer.fit_transform(sents_list)

print(x)

print(tfidf_vectorizer.get_feature_names())