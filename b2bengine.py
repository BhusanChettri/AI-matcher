from controller import Controller

class B2BEngine:
    """
    B2BEngine class for running the B2B recommendation system.
    """

    def __init__(self, config_file, field_names, feature_columns):
        """
        Initialize the B2BEngine class.

        Args:
            config_file (str): Path to the configuration file.
            field_names (list): List of field names for input data.
            feature_columns (list): List of feature columns.
        """
        self.config_file = config_file
        self.feature_columns = feature_columns
        self.field_names = field_names

    def run(self):
        """
        Run the B2B recommendation system.
        """
        controller = Controller(self.config_file, self.field_names, self.feature_columns)        
        controller.run()



if __name__ == "__main__":
        
    configuration_file = 'configuration/config.ini'        
    feature_columns = ["Industry", "Size", "Services", "Geography", "Remarks"]
    field_names = ["Name", "Industry", "Size", "Services", "Geography", "Certifications", "Remarks"]
    
    engine = B2BEngine(configuration_file, field_names, feature_columns)
    engine.run()