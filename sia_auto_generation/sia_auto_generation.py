import requests
import json
from sci_spacy.sci_spacy import SciSpacy
from nltk_word_net.nltk_word_net import NltkWordnet
from dataset_reader.dataset_reader import SiaDatasetReader


class SiaAutoGeneration():
    def __init__(self, config, cuda_device: False):
        query_generation_config = config['query_generation']
        sci_spacy_config = config['sci_spacy']
        dataset_reader_config = config['dataset_reader']

        self.query_generation_url = query_generation_config['url']
        self.sci_spacy = SciSpacy(sci_spacy_config, cuda_device)
        self.nltk_word_net = NltkWordnet()
        self.sia_dataset_reader = SiaDatasetReader(dataset_reader_config)

    def create_dataset(self, num_sia_generation):
        count = 0
        for sentence_4 in self.sia_dataset_reader.read_sentence_4():
            if (count == num_sia_generation):
                break
            self.create_data(sentence_4)
            count += 1

    def create_data(self, sentence_4):
        query = self.get_query(sentence_4)
        sentence_3 = self.get_sentence_3()
        sentence_2 = self.get_sentence_2()
        sentence_1 = self.get_sentence_1()

        json_data = {'query': query, 'sentence_4': sentence_4,
                     'sentence_3': sentence_3, 'sentence_2': sentence_2, 'sentence_1': sentence_1}
        json_data = json.dumps(json_data, indent=4, sort_keys=False)
        print(json_data)

        self.sia_dataset_reader.write_dataset(
            query, sentence_4, sentence_3, sentence_2, sentence_1)

    def get_query(self, sentence_4):
        # Obtain query using a Sentence 4 via generative model API
        payload = {'sentence': sentence_4}
        req = requests.get(self.query_generation_url, payload).json()
        query = req['query']
        return query

    def get_sentence_3(self):
        return ''

    def get_sentence_2(self):
        return ''

    def get_sentence_1(self):
        return ''
