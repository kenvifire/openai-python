from api import get_embedding
from utils import (
    cosine_similarity,
    download_nltk_data,
    preprocess_text
)
import os
import pandas as pd
import numpy as np

download_nltk_data()


dataset_file_path = os.path.join(
    os.path.dirname(__file__),
    'data',
    'simplified_coffee.csv'
)

input_coffee_name = input('Enter a coffee name:')
df = pd.read_csv(
    dataset_file_path,
    nrows=50
)

df['preprocessed_review'] = df['review'].apply(
    preprocess_text
)

model = 'text-embedding-ada-002'
review_embeddings = []
for review in df['preprocessed_review']:
    review_embeddings.append(
        get_embedding(
            review,
            model
        )
    )

try:
    input_coffee_index = df[df['name'] == input_coffee_name].index[0]
except:
    print("Please enter a valid coffee name.")
    exit()


similarities = []
input_review_embedding = review_embeddings[input_coffee_index]

for review_embedding in review_embeddings:
    similarity = cos_similarity(
        input_review_embedding,
        review_embedding
    )
    similarities.append(similarity)

most_similar_incides = np.argsort(similarities)[-6:-1]

similar_coffee_names = df.iloc[most_similar_incides]['name'].tolist()

print(
    "The most similr coffees to "
    f"{input_coffee_name} are:" )

for coffee_name in similar_coffee_names:
    print(coffee_name)
