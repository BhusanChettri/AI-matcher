from model import Model
from data_processor import DataProcessor
from feature_extractor import FeatureExtractor
import configparser

import pandas as pd


def train_models(config, feature_columns):
    """
    Trains models based on the provided configuration and feature columns.

    Args:
        config (configparser.ConfigParser): Configuration object containing the settings.
        feature_columns (list): List of feature columns to be used in training.

    Returns:
        None

    This function loads data, processes it, extracts features, and trains models based on the given configuration
    and feature columns. The trained models are saved for future use.

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

    print('Total number of features: %d' % (len(training_features)))

    model.train_and_save_models(training_features)


config_file = 'configuration/config.ini'
#field_names = ["Name", "Industry", "Size", "Services", "Geography", "Certifications", "Remarks"]
feature_columns = ["Industry", "Size", "Services", "Geography", "Remarks"]

train_models(config_file, feature_columns)