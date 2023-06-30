from model import Model
from data_processor import DataProcessor
from feature_extractor import FeatureExtractor
import configparser
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

import pandas as pd


def train_clusters_for_different_K(config, feature_columns):
    """
    Trains different KMeans using differnt K values to see which is the best for our data.

    Args:
        config (configparser.ConfigParser): Configuration object containing the settings.
        feature_columns (list): List of feature columns to be used in training.

    Returns:
        None

    """
    
    config = configparser.ConfigParser()
    config.read(config_file)

    data_processor = DataProcessor(config)
    feature_extractor = FeatureExtractor(config)
    model = Model(config)

    # path to raw data file is provided in config already
    dataFrames = data_processor.load_data()

    if 'Looking for' in dataFrames.columns:
        dataFrames = dataFrames.rename(columns={'Looking for': 'Services'})
    
    training_data = data_processor.process_data(dataFrames, feature_columns)

    feature_extractor.extract_features(training_data)  # FEATURES SAVED

    # Load the features
    training_features =  feature_extractor.load_features()
    
    compressor = model.train_compressor(training_features)
    cFeatures = model.compress_data(training_features, compressor)
    
    print('Total number of features: %d' % (len(training_features)))
    print(len(cFeatures[0]))
    
    # Lets try fitting the data for different values of k and record the inertia
    inertia_values = [] 
    for i in range(1, 11): 
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(cFeatures) 
        inertia_values.append(kmeans.inertia_)
    
    k_values = range(1, len(inertia_values) + 1)

    plt.plot(k_values, inertia_values, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Curve')

    # Save the plot as a PNG image
    plt.savefig('elbow_curve.png')
    
    
config_file = 'configuration/config.ini'
feature_columns = ["Industry", "Size", "Services", "Geography", "Remarks"]

train_clusters_for_different_K(config_file, feature_columns)