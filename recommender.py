import numpy as np
import torch
from sentence_transformers import util

class Recommender:
    def __init__(self, config):
        """
        Initialize the Recommender object.

        Args:
            config (ConfigParser): Configuration parser object.
        """
        self.n_points = config.getint('Test', 'top_n_results')
        self.similarity_metric = config.get('Test', 'similarity_metric')
    
    def compute_similarity(self, query_vector, vector_corpus):
        """
        Compute the similarity scores between a query vector and a vector corpus.

        Args:
            query_vector (ndarray): Query vector.
            vector_corpus (ndarray): Vector corpus.

        Returns:
            ndarray: Similarity scores.
        """
        if self.similarity_metric == 'dot_product':
            similarity = self.dot_product_similarity(query_vector, vector_corpus)
        else:
            similarity = self.cosine_similarity(query_vector, vector_corpus)
    
        return similarity
    
        
    def cosine_similarity(self, query_vector, corpus):
        """
        Compute cosine similarity scores between a query vector and a corpus of vectors.

        Args:
            query_vector (ndarray): Query vector.
            corpus (ndarray): Vector corpus.

        Returns:
            ndarray: Cosine similarity scores.
        """

        return util.cos_sim(query_vector, corpus)[0]
        
    def dot_product_similarity(self, query_vector, corpus):
        """
        Compute dot product similarity scores between a query vector and a corpus of vectors.

        Args:
            query_vector (ndarray): Query vector.
            corpus (ndarray): Vector corpus.

        Returns:
            ndarray: Dot product similarity scores.
        """
        pass  # TODO: Implement dot product similarity calculation
    

    def make_recommendations(self, query_vector, corpus):
        """
        Find the top N closest data points from the corpus to the given query vector.

        Args:
            query_vector (ndarray): Query vector.
            corpus: Corpus of embeddings to perform search. This is usually the cluster identified to have the closest match for the query vector.

        Returns:
            torch.Tensor: Top N closest data points (it has indices and scores of N data points)
        """
        corpus_embeddings = np.asarray(corpus)
        top_k = min(self.n_points, len(corpus))
        
        scores = self.compute_similarity(query_vector, corpus_embeddings)
                
        top_results = torch.topk(scores, k=top_k)
        
        # Return the top N closest data points
        return top_results

