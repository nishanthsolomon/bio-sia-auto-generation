import nltk
from nltk.corpus import wordnet as wn


class NltkWordnet():
    def __init__(self):
        nltk.download('wordnet')

    def get_synsets(self, text):
        return wn.synsets(text)

    def get_hypernyms(self, input_word):
        word = wn.synset(input_word)
        hypernyms = []
        for hypernym in word.hypernyms():
            hypernyms.append((hypernym.lemma_names(), hypernym.definition()))
        return hypernyms

    def get_hyponyms(self, input_word):
        word = wn.synset(input_word)
        hyponyms = []
        for hyponym in word.hyponyms():
            hyponyms.append((hyponym.lemma_names(), hyponym.definition()))
        return hyponyms
