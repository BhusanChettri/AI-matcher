import pandas as pd
from faker import Faker
import os
import random

fake = Faker()

# Options for service providers
provider_industry_options = ["Finance", "Marketing", "IT"]
provider_services_options = {
    "IT": [
        "Hardware sales",
        "Software sales",
        "End-user support",
        "Office setup",
        "Private cloud",
        "Managed services",
        "End-user support",
        "Cyber security services",
        "IT audit and compliance",
    ],
    "Finance": [
        "Accounting",
        "Audit",
        "Taxes",
        "Family office",
        "Stock Market",
        "Asset Management",
    ],
    "Marketing": ["Strategy marketing advisory"],
}
provider_geography_options = ["Western Europe", "Middle-East"]
provider_certifications_options = ["None", "ISO", "FINMA", "ISAE"]
provider_remarks_options = [
    "Good reviews",
    "Local presence",
    "More than 150 clients",
    "Multi-family office",
    "New company",
    "Result-oriented concept",
    "Expert in cloud computing",
    "Word of mouth until now",
    "Presence in data protected jurisdictions",
]

# Options for service consumers
consumer_industry_options = ["Finance", "Real Estate", "IT", "Healthcare"]
consumer_size_options = ["Small", "Medium", "Large"]
consumer_looking_for_options = [
    "Cloud hosting",
    "Digital Transformation",
    "Website revamp and digital identity refresh",
    "IT support",
    "Outsource billing services",
    "IT security hardening",
    "Introduction of security awareness",
]
consumer_geography_options = ["Middle East", "Western Europe"]
consumer_remarks_options = [
    "Established Bank",
    "Published an RFP",
    "Downloaded a guide on digital transformation",
    "Posted a request on LinkedIn",
    "Medical clinic with various health specialists",
    "Need identified in HubSpot",
    "Request on LinkedIn",
    "Downloaded a guide on cybersecurity services",
]

def generate_fake_dataset(n, output_path):
    
    """
    Generate fake datasets for service providers and service consumers and save them as CSV files.

    Args:
        n (int): Number of fake records to generate.
        output_path (str): Output directory path to save the generated datasets.

    Returns:
        None
    """
    
    # Generate fake data for service providers
    service_providers = []

    for _ in range(n):
        industry = random.choice(provider_industry_options)
        provider = {
            "Provider ID": fake.unique.random_number(digits=6),
            "Name": fake.company(),
            "Industry": industry,
            "Size": fake.random_element(elements=("Small", "Medium", "Large")),
            "Services": random.choice(provider_services_options[industry]),
            "Geography": random.choice(provider_geography_options),
            "Certifications": random.choice(provider_certifications_options),
            "Remarks": random.choice(provider_remarks_options),
        }
        service_providers.append(provider)
  
    # Generate fake data for service consumers
    service_consumers = []
    for _ in range(n):
        consumer = {
            "Consumer ID": fake.unique.random_number(digits=6),
            "Name": fake.company(),
            "Industry": random.choice(consumer_industry_options),
            "Size": random.choice(consumer_size_options),
            "Looking for": random.choice(consumer_looking_for_options),
            "Geography": random.choice(consumer_geography_options),
            "Remarks": random.choice(consumer_remarks_options),
        }
        service_consumers.append(consumer)

    # Create DataFrames from the generated data
    providers_df = pd.DataFrame(service_providers)
    consumers_df = pd.DataFrame(service_consumers)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define output file paths
    providers_output_file = os.path.join(output_path, "service_providers.csv")
    consumers_output_file = os.path.join(output_path, "service_consumers.csv")

    # Write the DataFrames to separate CSV files
    providers_df.to_csv(providers_output_file, index=False)
    consumers_df.to_csv(consumers_output_file, index=False)

    print("Fake datasets generated and saved to:", output_path)
    

# Set the number of fake records and the output path
n = 2000
output_path = 'fakedata-new'

# Call the function to generate the fake datasets
generate_fake_dataset(n, output_path)