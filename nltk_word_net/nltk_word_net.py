from nltk.corpus import wordnet as wn


class NltkWordnet():

    def get_synsets(self, text):
        return wn.synsets(text)

    def get_hypernyms(self, input_word):
        word = wn.synset(input_word)
        return word.hypernyms()

    def get_hyponyms(self, input_word):
        word = wn.synset(input_word)
        return word.get_hyponyms()
