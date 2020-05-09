from nltk.corpus import wordnet as wn


class NltkWordnet():
    def __init__(self):
        self.word = None

    def get_synsets(self, text):
        return wn.synsets(text)

    def set_word(self, input_word):
        self.word = wn.synset(input_word)

    def get_hypernyms(self):
        return self.word.hypernyms()

    def get_hyponyms(self):
        return self.word.get_hyponyms()
