from scipy import spatial

class Utilities:
    
    @staticmethod
    def get_cosine_similarity(embedding_1, embedding_2):
        cosine_similarity = 1 - spatial.distance.cosine(embedding_1, embedding_2)
        return cosine_similarity