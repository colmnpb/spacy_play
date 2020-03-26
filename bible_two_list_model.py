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

#list is 15 top "violent" terms handpicked from TF-IDF
list_a = ['burn','fall','die','destroy', 'smite', 'break','slay','fear','flee','kill','hate',
            'bury','weep','curse','slew']

list_b = ['jesus','god','pharoh']            

len_list_a = len(list_a)
len_list_b = len(list_b)

def find_matches_from_both_lists(spacy_doc, list_a, list_b):
  matches_a = set()
  matches_b = set()
  retval = {}

  for token in spacy_doc:
    if token.lemma_ in list_a:
      matches_a.add(token.lemma_)
    if token.lemma_ in list_b:
      matches_b.add(token.lemma_)

  if not matches_a or not matches_b:
    retval['matches_from_both'] = False
  else:
    retval['matches_from_both'] = True

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


def create_feature_array_from_matches(matches, possible_matches):
  X = [0] * len(possible_matches)
  for m in matches:
    m_index = possible_matches.index(m)
    X[m_index] = 1
  return X

# def remove_line_headers(line):
#   return re.sub("^\d?[A-Za-z]{1,5}\d*:\d*\w*","", line)

def parse_line_into_components(line):
  dict = {}
  #print(line)
  m = re.search('^(\d?[A-Za-z]{1,5})(\d*):(\d*)\s{1}(.*)$', line)
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

if __name__ == "__main__": 

  possible_matches = list(combine_matches(list_a, list_b))
  len_possible_matches = len(possible_matches)
  
  with open('/home/colmnpb/Downloads/kjv/kjv.txt', "r") as f:
    raw_text = f.readlines()

  #remove first line (header)
  raw_text.pop(0)

  # parse raw text into dicts
  parsed_lines = [parse_line_into_components(line) for line in raw_text]

  # get list of books (unordered)
  books = list(set([line['book'] for line in parsed_lines]))

  by_book = {}

  for book in books:
    by_book[book] = [line['text'] for line in parsed_lines if line['book'] == book]
    by_book[book] = "".join(by_book[book])


  nlp = spacy.load('en_core_web_lg')
  print("loaded spacy model")
  spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS


  X_sentences = []
  X_book = []
  X = []
  Y = []

  ## for testing only!
  # by_book = {}
  # by_book['nomatches'] = 'abc def gjea. flerf abfewafe barf.'
  # by_book['onlya'] = 'I weep with hate.  I smite with joy.'
  # by_book['onlyb'] = 'I believe there is a god.  Jesus was real.  There was a Pharoh.'
  # by_book['bookonlymatches'] = 'I fall off the steps. I talked to jesus'
  # by_book['bookonlytwomatches'] = 'I fall off the steps. I talked to jesus.  i broke my pencil'
  # by_book['sentmatches'] = 'I fell from jesus. god smite wicked. pharoh burn tar'
  # books = by_book.keys()
  for book in books:
    print(book)
    X_sentences = [0] * len_possible_matches
    X_book = [0] * len_possible_matches
    book_matches_a = set()
    book_matches_b = set()
    doc = nlp(by_book[book])
    for sent in doc.sents:    
      match_res = find_matches_from_both_lists(sent, list_a, list_b)
      if match_res['matches_from_both']:
          # set feature for sentence
          sentences_matches = combine_matches(match_res['matches_a'], match_res['matches_b'])
          X_sentences = create_feature_array_from_matches(sentences_matches, possible_matches)
      
      # add matches from this sentence to the matches for the book
      book_matches_a.update(match_res['matches_a'])
      book_matches_b.update(match_res['matches_b'])
    
    book_matches = combine_matches(book_matches_a, book_matches_b)
    X_book = create_feature_array_from_matches(book_matches, possible_matches)

    # combine matches found in all the sentences in the book and the matches for the book overall
    X.append(X_sentences + X_book)
    if (is_new_testament(book)):
      Y.append(1)
    else:
      Y.append(0)


  #test/train split

  X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3, random_state=42)

  # # simple DTC first (only 90 features, Random Forest would be overkill!)

  tree_model = DecisionTreeClassifier()
  tree_model.fit(X_train, Y_train)

  tree_model_predictions = tree_model.predict(X_test)

  cm = confusion_matrix(Y_test, tree_model_predictions)

  print(cm)

