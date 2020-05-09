import spacy


class SciSpacy():
    def __init__(self, config, cuda_device: False):

        if cuda_device:
            spacy.prefer_gpu()

        model_path = config['model_path']
        self.nlp = spacy.load(model_path)
        self.doc = None

    def get_doc(self, text):
        self.doc = self.nlp(text)

    def get_sentences(self):
        return list(self.doc.sents)

    def get_entities(self):
        entities = []

        for ent in self.doc.ents:
            entities.append(ent.text, ent.lemma_)

        return entities
