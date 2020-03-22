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

def find_matches_from_both_lists(doc, list_a, list_b):
  matches_a = set()
  matches_b = set()
  retval = {}

  for token in doc:
    if token.lemma_ in list_a:
      matches_a.add(token.lemma_)
    if token.lemma_ in list_b:
      matches_b.add(token.lemma_)

  if not matches_a or not matches_b:
    retval['matches_found'] = False
    retval['matches_a'] = set()
    retval['matches_b'] = set()
  else:
    retval['matches_found'] = True
    retval['matches_a'] = matches_a
    retval['matches_b'] = matches_b

  return retval

def combine_matches(list_a, list_b):

  combined_matches = set()

  for a_item in list_a:
    for b_item in list_b:
      if a_item < b_item:
        combined_matches.add(a_item + b_item)
      else:
        combined_matches.add(b_item + a_item)

  return combined_matches


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

### test 
# doc1 = nlp('The quick brown fox jumped over the lazy dog. I am a lazy developer. ')
# list_a = ['quick', 'brown', 'lazy', 'stick']
# list_b = ['fox','dog', 'airplane']

# t1 = find_matches_from_both_lists(doc1, list_a, list_b)
# print(t1)

# doc2 = nlp("Able was I ere I saw Elba. I'm a lazy developer, but can be quick at times.")
# t2 = find_matches_from_both_lists(doc2, list_a, list_b)
# print(t2)

# doc3 = nlp("My dog chased a fox while I was looking at an airplane.")
# t3 = find_matches_from_both_lists(doc3, list_a, list_b)
# print(t3)

list_a = set(['red', 'green','yellow','blue', 'red']) #extra red should be dropped by set
list_b = set(['dog','cat','cow','gnu'])

combined_matches = combine_matches(list_a,list_b)
print( combined_matches)

exit()

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

# print(X)
#print(Y)


#doc = nlp("".join(raw_lines))
#doc = nlp("".join(raw_lines[1:100]))

# verbs = {}
# for token in doc:

#   #print(token, "::", token.pos_)

#   if token.pos_ == "VERB":
#     print(token, "-->", token.lemma_)
#     if not token.lemma_ in verbs:
#       verbs[token.lemma_] = 1
#     else:
#       verbs[token.lemma_] = verbs[token.lemma_] + 1


# with open('verbs.csv', "w") as f:
#   for key in verbs:
#    f.write(key)
#    f.write(", ")
#    f.write( str(verbs[key]))
#    f.write("\n")


# sents_list = []
# for sent in doc.sents:
#     sents_list.append(sent.text)
# #print(sents_list)

# tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=100)

# x = tfidf_vectorizer.fit_transform(sents_list)

# print(x)

# print(tfidf_vectorizer.get_feature_names())

#test/train split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state=42)

# simple DTC first (only 15 features, Random Forest would be overkill!)

tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, Y_train)

tree_model_predictions = tree_model.predict(X_test)

cm = confusion_matrix(Y_test, tree_model_predictions)

print(cm)

