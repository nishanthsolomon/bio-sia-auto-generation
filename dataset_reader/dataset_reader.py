

class SiaDatasetReader():
    def __init__(self, config):
        self.read_path = config['read_path']
        self.write_path = config['write_path']

    def read_sentence_4(self):
        with open(self.read_path) as file_dataset:
            corpus = file_dataset.read().splitlines()
            for data in corpus:
                yield data
    
    def write_dataset(self, query, sentence_4, sentence_3, sentence_2, sentence_1):
        data = query + '\t' + sentence_4 + '\t' + sentence_3 + '\t' + sentence_2 + '\t' + sentence_1 + '\n'
        with open(self.write_path, 'a') as write_file:
            write_file.write(data)
