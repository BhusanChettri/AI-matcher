from sentence_transformers import SentenceTransformer
import os
import pickle
from typing import List
import numpy as np


class FeatureExtractor:
    def __init__(self, config):
        """
        Initialize the FeatureExtractor object.

        Args:
            config (ConfigParser): Configuration parser object.

        Attributes:
            transformerModelName (str): Name of the pre-trained transformer model.
            featExtractor: Pre-trained sentence transformer model for feature extraction.
        """
        self.featuresPath = config.get('FeatureExtractor', 'savePath')
        self.save_features = config.getboolean('FeatureExtractor', 'save_features')

        self.transformerModelName = config.get('FeatureExtractor', 'transformerModelName')
        self.featExtractor = self.load_feature_extractor()
        print('Loaded sentence-transformer for feature extraction..')

        if not os.path.exists(self.featuresPath):
            os.makedirs(self.featuresPath)
        self.featureFile = os.path.join(self.featuresPath, config.get('FeatureExtractor', 'featureFileName'))

    def load_feature_extractor(self):
        """
        Load the pre-trained sentence transformer model for feature extraction.

        Returns:
            SentenceTransformer: Pre-trained sentence transformer model.
        """        
        
        return SentenceTransformer(self.transformerModelName)

    def extract_feature(self, sentence: str):
        """
        Extract the feature embedding for a given sentence.

        Args:
            sentence (str): Input sentence.

        Returns:
            np.ndarray: Feature embedding vector.
        """
        return self.featExtractor.encode(sentence)
    
    def extract_features(self, sentence_list: List[str]):
        """
        Perform data preprocessing and extract feature embeddings for a list of sentences.

        Args:
            sentence_list (List[str]): List of sentences.

        Returns:
            List[np.ndarray]: List of feature embeddings.
        """
        features = []

        try:
            features = [self.extract_feature(sentence) for sentence in sentence_list]
            
            if self.save_features:
                with open(self.featureFile + '.pkl', 'wb') as file:
                    pickle.dump(features, file)
                print('Features saved in', self.featureFile)

            return features

        except Exception as e:
            print("Error occurred during feature extraction:", str(e))
            return []


    def load_features(self):
        """
        Returns the features saved in the savePath directory.
        """
        feat_file = self.featureFile + '.pkl'

        try:
            with open(feat_file, 'rb') as file:
                loaded_features = pickle.load(file)
                print('Feature file %s loaded..' % feat_file)
                return loaded_features
        except Exception as e:
            print("Error occurred while loading features:", str(e))
            return None
