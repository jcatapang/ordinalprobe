# src/utils.py
import numpy as np
import pandas as pd
from typing import Dict, List, Any

# For static embeddings
import gensim.downloader as api
from bpemb import BPEmb

# For transformer-based models
from transformers import AutoTokenizer, AutoModel
import torch

def load_ordinal_terms(filepath: str) -> Dict[str, List[str]]:
    """
    Loads ordinal terms from a CSV file.
    Each column is expected to represent an ordinal category.
    
    Preprocessing:
      - Drops missing values.
      - Strips whitespace from each term.
      - Filters out terms that are empty after stripping.
      - Lowercases all terms.
    """
    df = pd.read_csv(filepath)
    ordinal_terms = {}
    for col in df.columns:
        # Drop NA, strip whitespace, filter out empty strings, and lower-case each token.
        terms = df[col].dropna().apply(lambda x: x.strip()).tolist()
        terms = [t.lower() for t in terms if t.strip() != ""]
        ordinal_terms[col] = terms
    return ordinal_terms

def load_embedding_model(model_name: str) -> Any:
    """
    Loads a pre-trained embedding model using established libraries.
    
    Supported model names:
      - "Word2Vec": Uses 'word2vec-google-news-300' via Gensim.
      - "GloVe": Uses 'glove-wiki-gigaword-100' via Gensim.
      - "fastText": Uses 'fasttext-wiki-news-subwords-300' via Gensim.
      - "BPEmb": Loads BPEmb for English.
      - "BERT", "RoBERTa", "GPT-2": Loads corresponding transformer model using Hugging Face Transformers.
    """
    if model_name == "Word2Vec":
        print("Loading Word2Vec model via Gensim. (This may take a few minutes.)")
        return api.load("word2vec-google-news-300")
    elif model_name == "GloVe":
        print("Loading GloVe model via Gensim.")
        return api.load("glove-wiki-gigaword-100")
    elif model_name == "fastText":
        print("Loading fastText model via Gensim.")
        return api.load("fasttext-wiki-news-subwords-300")
    elif model_name == "BPEmb":
        print("Loading BPEmb model.")
        # Configure BPEmb as needed (here vs: vocabulary size; dim: embedding dimension).
        return BPEmb(lang="en", vs=100000, dim=300)
    elif model_name in ["BERT", "RoBERTa", "GPT-2"]:
        if model_name == "BERT":
            model_id = "bert-base-uncased"
        elif model_name == "RoBERTa":
            model_id = "roberta-base"
        elif model_name == "GPT-2":
            model_id = "gpt2"
        print(f"Loading {model_name} model from Hugging Face Transformers.")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id)
        # Set up device: GPU if available, otherwise CPU.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return {"tokenizer": tokenizer, "model": model, "device": device}
    else:
        raise ValueError(f"Model {model_name} not supported.")

def get_embeddings_for_terms(model: Any, terms: List[str], model_name: str) -> np.ndarray:
    """
    Retrieves embeddings for a list of terms from the given model.
    
    For BPEmb and Gensim models, we use their API.
    For transformer-based models, we compute the average token embedding per term.
    
    For Gensim-based models, if a full compound term is not found in the vocabulary,
    we try splitting it into tokens and averaging the embeddings of the constituent tokens.
    """
    embeddings = []
    missing_terms = []
    
    if model_name in ["Word2Vec", "GloVe", "fastText"]:
        for term in terms:
            if term in model.key_to_index:
                vec = model.get_vector(term)
            else:
                # Attempt to split the compound word and average the embeddings.
                tokens = term.split()
                token_vecs = []
                for token in tokens:
                    if token in model.key_to_index:
                        token_vecs.append(model.get_vector(token))
                if token_vecs:
                    vec = np.mean(token_vecs, axis=0)
                else:
                    missing_terms.append(term)
                    continue
            embeddings.append(vec)
    elif model_name == "BPEmb":
        for term in terms:
            try:
                # BPEmb.embed returns a 2D array; average over tokens.
                token_vecs = model.embed([term])[0]
                vec = np.mean(token_vecs, axis=0)
            except Exception:
                missing_terms.append(term)
                continue
            embeddings.append(vec)
    elif model_name in ["BERT", "RoBERTa", "GPT-2"]:
        # Transformer-based: use the tokenizer and model dictionary.
        tokenizer = model["tokenizer"]
        transformer_model = model["model"]
        device = model["device"]
        for term in terms:
            inputs = tokenizer(term, return_tensors="pt")
            # Move inputs to the correct device.
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = transformer_model(**inputs)
            # Get the last hidden state. Shape: [1, seq_length, hidden_dim]
            hidden_states = outputs.last_hidden_state.squeeze(0)
            # Convert token ids to tokens to filter out special tokens (if applicable).
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
            filtered_states = []
            for token, vec in zip(tokens, hidden_states):
                if model_name in ["BERT", "RoBERTa"]:
                    if token in tokenizer.all_special_tokens:
                        continue
                filtered_states.append(vec.cpu().numpy())
            if not filtered_states:
                missing_terms.append(term)
                continue
            avg_vec = np.mean(filtered_states, axis=0)
            embeddings.append(avg_vec)
    else:
        raise ValueError("Unsupported model type for embedding retrieval.")
    
    if missing_terms:
        print(f"Warning: The following terms were not found in {model_name}: {missing_terms}")
    return np.array(embeddings)
