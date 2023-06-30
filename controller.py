import configparser
from model import Model
from recommender import Recommender
from data_processor import DataProcessor
from feature_extractor import FeatureExtractor
from io_manager import IOManager
import pandas as pd
import streamlit as st

import configparser

class Controller:
    """
    Controller class for the B2B recommendation system.
    """

    def __init__(self, config_file, field_names, feature_columns):
        """
        Initialize the Controller class.

        Args:
            config_file (str): Path to the configuration file.
            field_names (list): List of field names for input data.
            feature_columns (list): List of feature columns.
        """
        self.config_file = config_file
        self.field_names = field_names
        self.feature_columns = feature_columns

        self.config = None
        self.data_processor = None
        self.feature_extractor = None
        self.recommender = None
        self.model = None

        self.training_features = None

    def initialize(self):
        """
        Initialize the necessary components for the B2B recommendation system.
        """
        self.load_config()
        
        self.data_processor = DataProcessor(self.config)
        self.feature_extractor = FeatureExtractor(self.config)
        self.model = Model(self.config)
        self.recommender = Recommender(self.config)
        self.training_features = self.feature_extractor.load_features()
        print('Initialization done.. app is ready..')

    def load_config(self):
        """
        Load the configuration file.
        """
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)

    def load_data(self):
        """
        Load the data.

        Returns:
            pandas.DataFrame: The loaded data.
        """
        dataFrames = self.data_processor.load_data()
        
        # to ensure field names are consistent
        if 'Looking for' in dataFrames.columns:
            dataFrames = dataFrames.rename(columns={'Looking for': 'Services'})
        return dataFrames

    def process_input(self, input_data):
        """
        Process the input data and perform recommendation.

        Args:
            input_data (pandas.DataFrame): The input data.
        """
        # Store the input_data as the next item in the data store - csv file/pandas

        # process test dataframe
        test_data = self.data_processor.process_data(input_data, self.feature_columns)

        # Extract features from the test data point
        test_features = self.feature_extractor.extract_feature(test_data)

        # Get the mostly likely cluster that matches the test data point
        cluster_id = self.model.predict_cluster(test_features)

        # Retrieve the training data points that form the cluster and perform nearest neighbor search on that
        # to obtain the closest matching data points
        cluster_data, data_indexes = self.model.get_cluster_data(cluster_id, self.training_features)
        result = self.recommender.make_recommendations(test_features, cluster_data)
        
        # Process the obtained results and pass it to the GUI
        processed_results = IOManager.process_result(result, data_indexes, self.load_data())
        IOManager.display_output(processed_results)
        
    def run(self):
        """
        Run the B2B recommendation system.
        """
        st.title("Service Provider Details")

        # Reads input entered and returns a DataFrame
        input_data = IOManager.read_input(self.field_names)
        
        if st.button("Submit"):
            
            # Initialize the models and necessary data points for obtaining model clusters
            self.initialize()
            
            # update the database by adding the newly obtained data point
            IOManager.update_database(input_data, self.config)
            
            # Process the input, perform feature extraction, and predict the most likely data points matching this
            self.process_input(input_data)
