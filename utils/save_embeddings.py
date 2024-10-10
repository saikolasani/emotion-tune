from openai import OpenAI 
client = OpenAI() 


import random # for testing suurposes 
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


"""
The goal of this code is to precompuate and save lexicon embeddings 
"""
#load VAD lookup table 

filepath = "NRC-VAD-Lexicon.txt" 

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

import json


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


def compute_and_save_lexicon_embeddings(lexicon, filepath="lexicon_embeddings_sample.json", model="text-embedding-ada-002", sample_fraction=0.01):
    """
    Compute embeddings for a sample of words in the lexicon and save to a file.
    
    Args:
    - lexicon: dict, the original VAD lexicon with words as keys and VAD vectors as values.
    - filepath: str, the path to save the lexicon embeddings.
    - model: str, the model to use for getting embeddings.
    - sample_fraction: float, fraction of the lexicon to sample (e.g., 0.02 for 2%).
    
    Returns:
    - None, saves the sampled embeddings to a JSON file.
    """
    # Sample 2% of the lexicon words
    total_words = list(lexicon.keys())
    sample_size = int(len(total_words) * sample_fraction)
    sampled_words = random.sample(total_words, sample_size)
    
    lexicon_embeddings = {}

    for i, word in enumerate(sampled_words):
        embedding = get_embedding(word, model=model)
        if embedding is not None:
            lexicon_embeddings[word] = embedding
        else:
            print(f"Failed to fetch embedding for '{word}'")
        
        # Add progress feedback (every 10 words for quick tests)
        if i % 10 == 0:
            print(f"Processed {i}/{sample_size} words")
    
    # Save the sampled lexicon embeddings to a file
    with open(filepath, "w") as f:
        json.dump(lexicon_embeddings, f)
    print(f"Sampled embeddings saved to {filepath}")

vad_lexicon = load_vad_lexicon(filepath)
compute_and_save_lexicon_embeddings(lexicon=vad_lexicon, filepath="lexicon_embeddings_sample.json", model="text-embedding-ada-002")