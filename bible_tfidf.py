import spacy
from spacy.lang.en import English
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
import re

v_lemmas = ['burn','fall','die','destroy', 'smite', 'break','slay','fear','flee','kill','hate',
            'bury','weep','curse','slew']

len_v_lemmas = len(v_lemmas)


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

def is_new_testament(book):
  nt_books = ['Mat','Mark', 'Luke','John','Acts','Rom','1Cor','2Cor', 'Gal', 'Eph','Phi', 'Col', '1Th', 
              '2Th', '1Tim','2Tim','Titus','Phmn','Heb','Jas','1Pet', '2Pet', '1Jn', '2Jn', '3Jn',
               'Jude', 'Rev']
  if book in nt_books:
    return True
  else:
    return False



with open('/home/colmnpb/Downloads/kjv/kjv.txt', "r") as f:
  raw_text = f.readlines()

#remove first line (header)
raw_text.pop(0)

#raw_lines = [remove_line_headers(line) for line in raw_text ]

parsed_lines = [parse_line_into_components(line) for line in raw_text]

# for line in parsed_lines:
#     print(line)

# get list of books (unordered)
books = list(set([line['book'] for line in parsed_lines]))

#print(books)

by_book = {}

for book in books:
  by_book[book] = [line['text'] for line in parsed_lines if line['book'] == book]
  by_book[book] = "".join(by_book[book])

# print
# print(type(by_book['Rev']))
# print

nlp = spacy.load('en_core_web_lg')
print("loaded spacy model")
spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS
# nlp.max_length = 5000000 #to handle entire text of bible

X = []
Y = []

for book in books:
  print(book)
  book_features = [0] * len_v_lemmas
  doc = nlp(by_book[book])
  for token in doc:
    for li, lemma in enumerate(v_lemmas):
      # print("token type:", type(token), "- > ", token)
      # print("lemma type", type(lemma), "- > ", lemma)
      if token.lemma_ == lemma:
        book_features[li] = 1
  #print(book_features)
  X.append(book_features)
  if (is_new_testament(book)):
    Y.append(1)
  else:
    Y.append(0)

print(X)
print(Y)





exit()

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


sents_list = []
for sent in doc.sents:
    sents_list.append(sent.text)
#print(sents_list)

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)

x = tfidf_vectorizer.fit_transform(sents_list)

print(x)

print(tfidf_vectorizer.get_feature_names())