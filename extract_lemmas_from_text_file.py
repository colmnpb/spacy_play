import sys
import spacy

if len(sys.argv) < 2:
  print("usage: python extract_lemmas_from_text_file.py <filename>")
  print()
  exit()

fn = sys.argv[1] 

with open(fn, "r") as f:
  raw_text = f.readlines()

nlp = spacy.load('en_core_web_sm')

text_doc = nlp("".join(raw_text))

text_lemmas = set()
for token in text_doc:
  if not(token.is_stop) and not(token.lemma_ == "\n") and  token.lemma_   :
      text_lemmas.add(token.lemma_)

with open('lemmas_from_' + fn, "w") as f:
  for lemma in text_lemmas:
      f.write(lemma)
      f.write("\n")
