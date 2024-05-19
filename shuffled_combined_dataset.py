# code 1:
import pandas as pd

# Load the data from the .pkl file
data = pd.read_pickle('/content/train.pkl')

# Define a function to extract the needed parts from the URLs
def extract_data(uri):
    # Get the last part after the last '/'
    return uri.split('/')[-1]

# Dictionary to hold the cleaned data
cleaned_data = {
    "subject": [],
    "predicate": [],
    "object": []
}

# Loop through each triple in the data
for subject, predicate, object_ in data:
    # Extract the useful part of each URI
    cleaned_data['subject'].append(extract_data(subject))
    cleaned_data['predicate'].append(extract_data(predicate))
    cleaned_data['object'].append(extract_data(object_))

# Turn the dictionary to a DataFrame
cleaned_df = pd.DataFrame(cleaned_data)

# Save the DataFrame to a CSV file
cleaned_df.to_csv('/content/cleaning_triples.csv', index=False)

#code 2:
import pandas as pd

# Load the cleaned data from the CSV file
cleaned_df = pd.read_csv('/content/cleaning_triples.csv')

# Dictionary to store pairs grouped by predicate
grouped = {}
for index, row in cleaned_df.iterrows():
    predicate = row['predicate']
    subject = row['subject']
    object_ = row['object']
    # If predicate not in dictionary, add it
    if predicate not in grouped:
        grouped[predicate] = []
    # Append the (subject, object) pair to the predicate
    grouped[predicate].append((subject, object_))

# Generate analogies based on common predicates
analogies = []
for predicate, pairs in grouped.items():
    #Only make analogies if there are at least two pairs for a predicate
    if len(pairs) > 1:
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                a, b = pairs[i]
                c, d = pairs[j]
                # Add the analogy to the list
                analogies.append({
                    "Entity1": a,
                    "Entity2": b,
                    "Entity3": c,
                    "Entity4": d,
                    "Label": 1  # Valid analogy
                })

# Convert the list of analogies to a DataFrame
analogies_df = pd.DataFrame(analogies)

# Save the analogies to a CSV file
analogies_df.to_csv('/content/valid_analogies.csv', index=False)

# Display the first few rows of the analogies DataFrame
analogies_df.head()

# code 3:
import pandas as pd
import random

# Load the data from the CSV file
data = pd.read_csv('/content/cleaning_triples.csv')

# Copy the data into a new DataFrame to work with
triples_df = data.copy()

# Function to corrupt a column with random replacement from the same column
def corrupt_column(column):
    # Replace each value in the column with a random one from the same column
    return [random.choice(triples_df[column].tolist()) for _ in range(len(triples_df))]

# Apply random replacement to each part with a random choice
for index, row in triples_df.iterrows():
    part_to_corrupt = random.choice(['subject', 'predicate', 'object'])  # Choose which part to corrupt
    if part_to_corrupt == 'subject':
        triples_df.at[index, 'subject'] = random.choice(triples_df['subject'].tolist())
    elif part_to_corrupt == 'predicate':
        triples_df.at[index, 'predicate'] = random.choice(triples_df['predicate'].tolist())
    else:
        triples_df.at[index, 'object'] = random.choice(triples_df['object'].tolist())

# Save the corrupted triples back to a CSV file
triples_df.to_csv('/content/corrupted_triples.csv', index=False)

#code 4:
import pandas as pd
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the CSV file
df = pd.read_csv('/content/valid_analogies.csv')

# Shuffle the rows of the DataFrame
shuffled_df = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled DataFrame to Google Drive
shuffled_df.to_csv('/content/drive/My Drive/shuffled_valid_analogies.csv', index=False)

print("The rows have been shuffled and saved to shuffled_valid_analogies.csv in your Google Drive")

#code 5:
import pandas as pd
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the cleaned data from the CSV file
cleaned_df = pd.read_csv('/content/corrupted_triples.csv')

# Create a dictionary to store pairs grouped by predicate
grouped = {}
for index, row in cleaned_df.iterrows():
    predicate = row['predicate']
    subject = row['subject']
    object_ = row['object']

    if predicate not in grouped:
        grouped[predicate] = []
    grouped[predicate].append((subject, object_))

# Generate analogies based on common predicates
analogies = []
for predicate, pairs in grouped.items():
    # Only generate analogies if there are at least two pairs for a predicate
    if len(pairs) > 1:
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                a, b = pairs[i]
                c, d = pairs[j]
                # Append the analogy to the list
                analogies.append({
                    "Entity1": a,
                    "Entity2": b,
                    "Entity3": c,
                    "Entity4": d,
                    "Label": 0  # Valid analogy
                })

# Convert the list of analogies to a DataFrame
analogies_df = pd.DataFrame(analogies)

# Save the analogies to a CSV file in Google Drive
analogies_df.to_csv('/content/drive/My Drive/invalid_analogies.csv', index=False)

# Display the first few rows of the analogies DataFrame
analogies_df.head()

# code 6:
import pandas as pd
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load the CSV file
df = pd.read_csv('/invalid_analogies (1).csv')

# Shuffle the rows of the DataFrame
shuffled_df = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled DataFrame to Google Drive
shuffled_df.to_csv('/content/drive/My Drive/shuffled_invalid_analogies.csv', index=False)

print("The rows have been shuffled and saved to shuffled_invalid_analogies.csv in your Google Drive")

# code 7:
from google.colab import drive
import pandas as pd


drive.mount('/content/drive')

# Load the dataset
file_path = '/content/drive/MyDrive/shuffled_valid_analogies.csv'  # Update this path
df = pd.read_csv(file_path)

# Sample the dataset
df_sampled = df.sample(n=500000, random_state=42)

# Save the reduced dataset
save_path = '/content/drive/MyDrive/data/reduced_valid_dataset.csv'  # Update this path if needed
df_sampled.to_csv(save_path, index=False)

print("The reduced dataset has been saved successfully!")

# code 8:
from google.colab import drive
import pandas as pd


drive.mount('/content/drive')

# Load the dataset
file_path = '/content/drive/MyDrive/shuffled_invalid_analogies.csv'  # Update this path
df = pd.read_csv(file_path)

# Sample the dataset
df_sampled = df.sample(n=500000, random_state=42)

# Save the reduced dataset
save_path = '/content/drive/MyDrive/data/reduced_invalid_dataset.csv'  # Update this path if needed
df_sampled.to_csv(save_path, index=False)

print("The reduced dataset has been saved successfully!")

#code 9 :
from google.colab import drive
import pandas as pd


drive.mount('/content/drive')

# Load the datasets
path_valid = '/content/drive/MyDrive/data/reduced_valid_dataset.csv'
path_invalid = '/content/drive/MyDrive/data/reduced_invalid_dataset.csv'
df_valid = pd.read_csv(path_valid)
df_invalid = pd.read_csv(path_invalid)

# Combine the datasets
df_combined = pd.concat([df_valid, df_invalid], ignore_index=True)

# Shuffle the dataset
df_combined = df_combined.sample(frac=1).reset_index(drop=True)

# Save the combined dataset
combined_save_path = '/content/drive/MyDrive/data/combined_dataset.csv'
df_combined.to_csv(combined_save_path, index=False)

print("The combined dataset has been saved successfully to Google Drive.")

# code 10 :

from google.colab import drive
import pandas as pd


drive.mount('/content/drive')

# Load the dataset
path_to_dataset = '/content/drive/MyDrive/data/combined_dataset.csv'
df = pd.read_csv(path_to_dataset)

# Shuffle the dataset
df_shuffled = df.sample(frac=1).reset_index(drop=True)

# Save the shuffled dataset
save_path = '/content/drive/My Drive/shuffled_combined_dataset.csv'
df_shuffled.to_csv(save_path, index=False)

print("Shuffled dataset saved successfully!")



