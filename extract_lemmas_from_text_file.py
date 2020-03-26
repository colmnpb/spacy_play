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

text_lemmas = {}
for token in text_doc:
  if token.lemma_ and not(token.lemma_ == "\n") :
      text_lemmas[token.lemma_] = "noop"

with open('lemmas_from_' + fn, "w") as f:
  for lemma in text_lemmas.keys():
      f.write(lemma)
      f.write("\n")
