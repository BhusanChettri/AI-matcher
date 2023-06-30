import os
import spacy
import pandas as pd
import string


class DataProcessor:
    def __init__(self, config):
        """
        Initialize the DataProcessor.

        Args:
            config (ConfigParser): Configuration object containing the necessary parameters.
        """
        self.pre_process = config.getboolean('DataProcessor', 'apply_preprocessing')
        self.nlpModelPath = config.get('DataProcessor', 'nlpModelPath')
        self.dataFilePath = config.get('DataProcessor', 'dataFilePath')
        self.nlpModel = None

        if self.pre_process:
            self.nlpModel = self.load_nlp_model()
            if self.nlpModel is None:
                raise RuntimeError("Failed to load Spacy NLP model. Preprocessing cannot continue.")

    def load_nlp_model(self):
        """
        Load the Spacy NLP model for text processing.

        Returns:
            spacy.Language: Loaded Spacy NLP model.
        """
        model_path = os.path.join(os.getcwd(), self.nlpModelPath)

        try:
            nlp_model = spacy.load(model_path)
            return nlp_model

        except OSError as e:
            print(f"Error loading Spacy NLP model from path: {model_path}")
            print(f"Exception: {e}")
            return None

    def load_data(self):
        """
        Load the data from a CSV file into a pandas DataFrame.

        Returns:
            pd.DataFrame: Loaded dataframe.
        """
        try:
            dataFrame = pd.read_csv(self.dataFilePath)
            return dataFrame

        except FileNotFoundError:
            print("File not found. Please make sure the file exists.")

        except Exception as e:
            print("An error occurred:", str(e))

    def process_data(self, dataFrame, useColumns):
        """
        Process the DataFrame by removing unwanted feature columns.

        Args:
            dataFrame (pd.DataFrame): The data on which processing is to be applied.
            useColumns (list): A list of column names to be used.

        Returns:
            list: Processed list of sentences.
        """
        sentence_list = []

        # Drop columns/features not present in useColumns
        df = self.filter_data_columns(dataFrame, useColumns)

        if df is not None:
            for _, row in df.iterrows():
                sentence = ' '.join(row.values.astype(str))

                if self.pre_process:
                    sentence = self.tokenize_text(sentence)

                sentence_list.append(sentence)

        return sentence_list
    
    

    def filter_data_columns(self, data, useColumns):
        """
        Filter the provided columns from the given dataframe.

        Args:
            data (pd.DataFrame): The dataframe to filter.
            useColumns (list): A list of column names to be used.

        Returns:
            pd.DataFrame: The filtered dataframe.
        """
        return data[useColumns]

    def tokenize_text(self, sentence):
        """
        Tokenize the input sentence using the pre-built Spacy English language model.

        Args:
            sentence (str): The input sentence to tokenize.

        Returns:
            str: Tokenized sentence.
        """
        
        # extract set of stop words and punctuations that do not carry much useful information for modelling
        stop_words = self.nlpModel.Defaults.stop_words
        punctuations = string.punctuation

        # get an doc instance from the spacy model
        doc = self.nlpModel(sentence)

        # clean tokesn by reducing each word to its root word and remove whitespaces
        # ignore words found in stop_words and punctuations        
        mytokens = [word.lemma_.lower().strip() for word in doc]
        mytokens = [word for word in mytokens if word not in stop_words and word not in punctuations]

        return ' '.join(mytokens)

    def count_tokens(self, sentence):
        """
        Count the number of tokens in a preprocessed sentence.

        Args:
            sentence (str): The preprocessed sentence.

        Returns:
            int: Number of tokens.
        """
        return len(sentence)

