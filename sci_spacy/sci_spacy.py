import spacy


class SciSpacy():
    def __init__(self, config, cuda_device: False):

        if cuda_device:
            spacy.prefer_gpu()

        model_path = config['model_path']
        self.nlp = spacy.load(model_path)

    def get_sentences(self, text):
        doc = self.nlp(text)
        return list(doc.sents)

    def get_entities(self, text):
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append((ent.text, ent.lemma_.replace(' ','_')))

        return entities
