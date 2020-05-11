import requests
import json
import truecase
from sci_spacy.sci_spacy import SciSpacy
from nltk_word_net.nltk_word_net import NltkWordnet
from dataset_reader.dataset_reader import SiaDatasetReader
from biobert_embedding.embedding import BiobertEmbedding
from utils.utils import Utilities


class SiaAutoGeneration():
    def __init__(self, config, cuda_device: False):
        query_generation_config = config['query_generation']
        sci_spacy_config = config['sci_spacy']
        dataset_reader_config = config['dataset_reader']

        self.query_generation_url = query_generation_config['url']
        self.sci_spacy = SciSpacy(sci_spacy_config, cuda_device)
        self.nltk_word_net = NltkWordnet()
        self.sia_dataset_reader = SiaDatasetReader(dataset_reader_config)
        self.biobert = BiobertEmbedding()

    def create_dataset(self, num_sia_generation):
        count = 0
        for sentence_4 in self.sia_dataset_reader.read_sentence_4():
            if (count == num_sia_generation):
                break
            self.create_data(sentence_4)
            count += 1

    def create_data(self, sentence_4):
        query = self.get_query(sentence_4)
        entities_replacements = self.get_entities_replacements(sentence_4)
        sentence_3 = truecase.get_true_case(self.get_sentence_3(sentence_4, entities_replacements))
        sentence_2 = truecase.get_true_case(self.get_sentence_2(sentence_3, entities_replacements))
        sentence_1 = truecase.get_true_case(entities_replacements[0][2])

        json_data = {'query': query, 'sentence_4': sentence_4,
                     'sentence_3': sentence_3, 'sentence_2': sentence_2, 'sentence_1': sentence_1}
        json_data = json.dumps(json_data, indent=4, sort_keys=False)
        print(json_data)

        self.sia_dataset_reader.write_dataset(
            query, sentence_4, sentence_3, sentence_2, sentence_1)

    def get_query(self, sentence_4):
        payload = {'sentence': sentence_4}
        req = requests.get(self.query_generation_url, payload).json()
        query = req['query']
        return query

    def get_entities_replacements(self, sentence_4):
        sentence_4_vector = self.biobert.sentence_vector(sentence_4)
        entities = self.sci_spacy.get_entities(sentence_4)

        entities_replacements = []

        for entity, lemma in entities:
            synsets = self.nltk_word_net.get_synsets(lemma)

            if(len(synsets) > 0):

                if (len(synsets) == 1):
                    select_synset = synsets[0].name()
                elif(len(synsets) > 1):
                    synset_names = []
                    cosine_similarities = []

                    for synset in synsets:
                        synset_definition_embedding = self.biobert.sentence_vector(
                            synset.definition())
                        cosine_similarity = Utilities.get_cosine_similarity(
                            sentence_4_vector, synset_definition_embedding)
                        synset_names.append(synset.name())
                        cosine_similarities.append(cosine_similarity)

                    select_synset = synset_names[cosine_similarities.index(
                        max(cosine_similarities))]

                hypernyms = self.nltk_word_net.get_hypernyms(select_synset)
                hyponyms = self.nltk_word_net.get_hyponyms(select_synset)

                select_hyponym = None
                select_hypernym = None
                if hyponyms:
                    cosine_similarities = []
                    for hyponym in hyponyms:
                        hyponym_definition_embedding = self.biobert.sentence_vector(
                            hyponym[1])
                        cosine_similarity = Utilities.get_cosine_similarity(
                            sentence_4_vector, hyponym_definition_embedding)
                        cosine_similarities.append(cosine_similarity)
                    select_hyponym = hyponyms[cosine_similarities.index(
                        max(cosine_similarities))][0][0]
                    select_hyponym_definition = hyponyms[cosine_similarities.index(
                        max(cosine_similarities))][1]
                    entities_replacement = (
                        entity, select_hyponym, select_hyponym_definition)

                if not select_hyponym:
                    if hypernyms:
                        cosine_similarities = []
                        for hypernym in hypernyms:
                            hypernym_definition_embedding = self.biobert.sentence_vector(
                                hypernym[1])
                            cosine_similarity = Utilities.get_cosine_similarity(
                                sentence_4_vector, hypernym_definition_embedding)
                            cosine_similarities.append(cosine_similarity)
                        select_hypernym = hypernyms[cosine_similarities.index(
                            max(cosine_similarities))][0][0]
                        select_hypernym_definition = hypernyms[cosine_similarities.index(
                            max(cosine_similarities))][1]
                        entities_replacement = (
                            entity, select_hypernym, select_hypernym_definition)

                entities_replacements.append(entities_replacement)

        return entities_replacements

    def get_sentence_3(self, sentence_4, entities_replacements):
        entities_replacement = entities_replacements[0]
        sentence_3 = sentence_4.replace(
            entities_replacement[0], entities_replacement[1].replace('_', ' '))
        return sentence_3

    def get_sentence_2(self, sentence_3, entities_replacements):
        sentence_2 = sentence_3
        for entities_replacement in entities_replacements[1:]:
            sentence_2 = sentence_2.replace(
                entities_replacement[0], entities_replacement[1].replace('_', ' '))
        return sentence_2
