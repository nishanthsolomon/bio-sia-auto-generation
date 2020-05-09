import requests
from sci_spacy.sci_spacy import SciSpacy
from nltk_word_net.nltk_word_net import NltkWordnet


class SiaAutoGeneration():
    def __init__(self, config, cuda_device: False):
        query_generation_config = config['query_generation']
        sci_spacy_config = config['sci_spacy']

        self.query_generation_url = query_generation_config['url']
        self.sci_spacy = SciSpacy(sci_spacy_config, cuda_device)
        self.nltk_word_net = NltkWordnet()

    def create_data(self, sentence_4):
        query = self.get_query(sentence_4)

        return query

    def get_query(self, sentence_4):
        # Obtain query using a Sentence 4 via generative model API
        payload = {'sentence': sentence_4}
        req = requests.get(self.query_generation_url, payload).json()
        query = req['query']
        return query

    def get_sentence_3(self):
        pass

    def get_sentence_2(self):
        pass

    def get_sentence_1(self):
        pass
