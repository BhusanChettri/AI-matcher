from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import numpy as np
import os
import pickle

class Model:
    def __init__(self, config):
        """
        Initialize the clustering model.

        Args:
            config (ConfigParser): Configuration parser object containing model parameters.
        """
        self.n_clusters = config.getint('Model', 'num_of_clusters')
        self.n_components = config.getint('Model', 'num_of_PCA_components')
        self.method = config.get('Model', 'clustering_method')
        self.distance_threshold = config.getfloat('Model', 'distance_threshold')
        self.compressData = config.getboolean('Model', 'compressData')

        self.saveModel = config.getboolean('Model', 'saveModel')
        self.savePath = config.get('Model', 'savePath')

        self.clusterName = config.get('Model', 'clusterModel')
        self.compressorName = config.get('Model', 'compressorModel')

        self.compressorModel = None

    def save_model(self, model, modelName):
        """
        Save the model to a file.

        Args:
            model: Model object to be saved.
            modelName (str): Name of the model.
        """
        savePath = os.path.join(self.savePath, modelName)

        if not os.path.exists(savePath):
            os.makedirs(savePath)

        modelFile = os.path.join(savePath, modelName + '.pkl')
        
        try:
            with open(modelFile, 'wb') as file:
                pickle.dump(model, file)
                print('Model saved in %s' % modelFile)
        except Exception as e:
            print("Error saving the model:", str(e))

    def load_model(self, modelName):
        """
        Load the model from a file.

        Args:
            modelName (str): Name of the model.

        Returns:
            The loaded model.
        """
        savePath = os.path.join(self.savePath, modelName)
        modelFile = os.path.join(savePath, modelName + '.pkl')

        try:
            with open(modelFile, 'rb') as file:
                loaded_model = pickle.load(file)
                return loaded_model
        except Exception as e:
            print("Error occurred while loading model:", str(e))
            return None

    def train_and_save_models(self, features):
        """
        Train and save the clustering and compressor models.

        Args:
            features (array-like): Feature embeddings used for training.
        """
        compressor = None

        if self.compressData:
            compressor = self.train_compressor(features)
            features = self.compress_data(features, compressor)

            # Save the compressor model
            self.save_model(compressor, self.compressorName)

        clusterModel = self.train_clusters(features)

        # Save the cluster model
        self.save_model(clusterModel, self.clusterName)

    def train_clusters(self, features):
        """
        Train the clustering model on the provided feature embeddings.

        Args:
            features (array-like): Feature embeddings used for clustering.

        Returns:
            The trained clustering model.
        """
        print('Training clustering model...')

        if self.method == 'Agglomerative':
            features = features / np.linalg.norm(features, axis=1, keepdims=True)
            
            # ensure n_clusters is set to None if distance_threshold has been set
            if self.distance_threshold is not None:
                self.n_clusters = None
                
            model = AgglomerativeClustering(n_clusters=self.n_clusters, distance_threshold=self.distance_threshold)
        else:
            model = KMeans(n_clusters=self.n_clusters)

        return model.fit(features)

    def predict_cluster(self, feature):
        """
        Predict the cluster for a given feature embedding.

        Args:
            feature (array-like): Feature embedding to be predicted.

        Returns:
            int: Predicted cluster label.
        """
        if self.compressData:
            compressor = self.load_model(self.compressorName)
            feature = self.compress_data(feature, compressor)

        clusterModel = self.load_model(self.clusterName)

        return clusterModel.predict(feature)[0]

    def train_compressor(self, features):
        """
        Train a PCA model for compressing the feature dimension.

        Args:
            features (array-like): Feature embeddings used for training the PCA model.

        Returns:
            object: Trained PCA model.
        """
        print('Training compressor model...')

        compressor = PCA(n_components=self.n_components)
        return compressor.fit(features)

    def compress_data(self, features, compressor):
        """
        Compress the feature embeddings using the trained PCA model.

        Args:
            features (array-like): Feature embeddings to be compressed.
            compressor: Trained PCA model.

        Returns:
            array-like: Compressed feature embeddings.
        """
        return compressor.transform(features)

    def get_cluster_data(self, clusterID, training_features):
        """
        Get all examples/features that belong to a specific cluster.

        Args:
            clusterID (int): Cluster ID.
            training_features (array-like): Training features used for clustering.

        Returns:
            array-like: Features belonging to the specified cluster.
        """
        clusterModel = self.load_model(self.clusterName)

        cluster_labels = clusterModel.labels_
        indexes = np.where(cluster_labels == clusterID)[0]
        features_in_cluster = np.asarray(training_features)[indexes]

        return features_in_cluster, indexes

    def retrain_models(self, featurePath):
        """
        Retrain the clustering and compressor models when new data is available.

        Args:
            featurePath (string): Path of the newly obtained features/data.

        Returns:
            Newly trained model.
        """
        pass
