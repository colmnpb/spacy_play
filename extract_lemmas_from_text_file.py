import spacy
#from spacy.lang.en import English

with open('raw_adj.txt', "r") as f:
  raw_adjs = f.readlines()

nlp = spacy.load('en_core_web_lg')

adj_doc = nlp("".join(raw_adjs))

adj_lemmas = {}
for token in adj_doc:
  if token.lemma_ and not(token.lemma_ == "\n") :
      print("->", token.lemma_, "<-")
      adj_lemmas[token.lemma_] = "noop"

with open("adj_lemmas.txt", "w") as f:
  for lemma in adj_lemmas.keys():
      print(lemma)
      f.write(lemma)
      f.write("\n")
