import pandas as pd
import streamlit as st
import os

class IOManager:
    
    @staticmethod
    def update_database(data_point, config):
        """
        Updates the database file - which in our case is a simple csv file
        
        Args:
            data_point: a pandas dataframe consisting of the data point to be saved in the database
            
        Returns:
            None
        """
        
        output_path = config.get('DataProcessor', 'ontheFlyDataPath')
        
        if os.path.isfile(output_path):
            data_point.to_csv(output_path, mode='a', header=False, index=False)
        else:
            data_point.to_csv(output_path, index=False)
    
    
    @staticmethod
    def read_input(field_names):
        """
        Reads the input fields for service provider details from the user.
        
        Args:
            field_names (list): List of field names as input.
        
        Returns:
            pandas.DataFrame: DataFrame object containing the input fields.
        """
        input_fields = {}
        for field in field_names:
            input_fields[field] = st.text_input(field)
        
        return pd.DataFrame([input_fields])


    @staticmethod
    def process_result(results, data_indexes, corpus):
        """
        Process the results and return a DataFrame with the specified columns.
        
        Args:
            results (list): List of processed results.
            data_indexes (list): List of original data indexes.
            corpus (pandas.DataFrame): Original corpus of data.
        
        Returns:
            pandas.DataFrame: Processed results as a DataFrame.
        """
        processed_results = []
        
        use_columns = corpus.columns

        for i, score in zip(results.indices, results.values):
            original_index = data_indexes[i]
            data_point = corpus.iloc[original_index]
            result = {}

            for column in use_columns:
                result[column] = data_point[column]

            result['Confidence Score'] = f'{score.item() * 100:.2f}%'
            processed_results.append(result)

        return processed_results

    @staticmethod
    def display_output(result):
        """
        Displays the list of potential clients as the output.
        
        Args:
            result (list): List of processed results as dictionaries.
        """
        st.title("Potential Clients")
            
        # Convert the processed results to a DataFrame
        df = pd.DataFrame(result, index=range(1, len(result) + 1))

        # Display the DataFrame as a table using Streamlit
        st.table(df)
