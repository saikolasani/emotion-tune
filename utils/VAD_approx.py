from openai import OpenAI 
client = OpenAI() 


import json
import random # for testing suurposes 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import os

"""
The goal of this code is to precompuate and save lexicon embeddings 
"""
#load VAD lookup table 

filepath = "utils/NRC-VAD-Lexicon.txt" 
def load_vad_lexicon(filepath):
    lexicon = {} 
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            try: 
                line = line.strip().split('\t') # use tab as the delimeter
                word = line[0]
                
                # #add a debug print statement to see the word 
                # print(word)
                
                vad =np.array(line[1:], dtype=float) # conver to float as the VAD values are float 
                lexicon[word] = vad
            except ValueError as e:
                print(f"Error at line {i}: {line}")
                continue
    return lexicon



def get_embedding(text, model="text-embedding-ada-002"):
    """
    Fetches the embedding for the input text using OpenAI's API.
    
    Args:
    text (str): The input text to fetch the embedding for.
    model (str): The OpenAI model name to use for fetching embeddings.
    
    Returns:
    list: Embedding vector for the input text or None if there's an error.
    """
    try:
        # Clean the text by replacing newline characters
        text = text.replace("\n", " ")

        # Fetch embedding from OpenAI
        response = client.embeddings.create(input=[text], model=model)
        
        # Access the embedding correctly based on the response structure
        embedding = response.data[0].embedding
        return embedding
    
    except Exception as e:
        print(f"Failed to fetch embedding for '{text}': {e}")
        return None



def load_lexicon_embeddings(filepath_lexicon_embeddings):
    """
    Load the precomputed lexicon embeddings from a file.
    
    Args:
    - filepath_lexicon_embeddings: str, the path to the saved lexicon embeddings.
    
    Returns:
    - lexicon_embeddings: dict, with words as keys and their embeddings as values.
    """
    with open(filepath_lexicon_embeddings, "r") as f:
        lexicon_embeddings = json.load(f)
    return lexicon_embeddings




def VAD_with_embeddings(w, lexicon_embeddings, lexicon, N=5):
    """
    Returns the VAD vector for a word `w` using precomputed embeddings.
    
    Args:
    - w: word to compute VAD for.
    - lexicon_embeddings: dict, precomputed embeddings for lexicon words.
    - lexicon: dict, the original VAD lexicon for fetching VAD values.
    - N: top N closest words.
    
    Returns:
    - The weighted average VAD vector for the word `w`.
    """
    print(f"Looking up word: {w}")
    
    if w in lexicon:
        print(f"Word '{w}' found in lexicon. VAD vector: {lexicon[w]}")
        return lexicon[w]  # If the word is in the lexicon, return its VAD

    # Step 1: Get the embedding for the input word `w`
    embedding_w = get_embedding(w, model="text-embedding-ada-002")
    if embedding_w is None:
        print(f"Embedding not found for word '{w}'.")
        return None

    # Step 2: Compute cosine similarity between `w` and all lexicon words
    vad_words = list(lexicon_embeddings.keys())
    vad_embeddings = np.array([lexicon_embeddings[word] for word in vad_words])
    
    similarities = cosine_similarity([embedding_w], vad_embeddings)[0]
    print(f"Similarities calculated: {similarities[:5]}")

    # Step 3: Get top-N closest words based on similarities
    top_n_indices = np.argsort(similarities)[-N:]
    top_n_words = [vad_words[i] for i in top_n_indices]
    print(f"Top-N closest words: {top_n_words}")
    
    # Step 4: Calculate weighted average of VAD vectors for top-N similar words
    top_n_similarities = similarities[top_n_indices]
    weighted_vad = np.average([lexicon[word] for word in top_n_words], axis=0, weights=top_n_similarities)
    
    return weighted_vad



vad_lexicon = load_vad_lexicon(filepath) 
file_pathlexicon_embeddings = "utils/lexicon_embeddings_sample.json"
# Check if the file exists
if os.path.exists(file_pathlexicon_embeddings):
    print(f"File found at {file_pathlexicon_embeddings}")
else:
    print(f"File not found at {file_pathlexicon_embeddings}")

#lexicon_embeddings = load_lexicon_embeddings(file_pathlexicon_embeddings) 
#vad_vector = VAD_with_embeddings("Happy", lexicon_embeddings, vad_lexicon, 5)
#print(f"Final VAD vector for 'Happy': {vad_vector}")