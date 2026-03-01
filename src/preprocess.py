"""
Data Preprocessing Module
=========================
Owner: Member 1 (Data Pipeline)

Responsibilities:
- Load raw DGA and benign domain CSVs
- Clean, deduplicate, and lowercase domains
- Strip TLD suffixes using tldextract
- Sort alphabetically (preserves Levenshtein sequence ordering)
- Generate English dictionary file from NLTK corpus
- Compute trigram frequency table from benign domain corpus
- Create stratified 80/20 train/test split (random_state=42)
"""

import os
import pandas as pd
import numpy as np
import nltk
import pickle
import tldextract
from sklearn.model_selection import train_test_split


def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load and combine DGA + benign domain CSVs into a single DataFrame.

    Returns:
        DataFrame with columns: ['domain', 'label']
        label: 1 = DGA (malicious), 0 = Normal (benign)
    """
    # TODO: Implement
    raise NotImplementedError


def strip_tld(domain: str) -> str:
    """Remove TLD suffix from domain using tldextract.

    Example: 'evil-domain.co.uk' -> 'evil-domain'
    """
    # TODO: Implement
    raise NotImplementedError


def clean_domains(df: pd.DataFrame) -> pd.DataFrame:
    """Clean domain DataFrame: lowercase, strip TLD, deduplicate, sort.

    Returns:
        Cleaned DataFrame sorted alphabetically by domain.
    """
    # TODO: Implement
    raise NotImplementedError


def build_english_dictionary(output_path: str) -> set:
    """Download NLTK words corpus and save as a text file.

    Returns:
        Set of lowercase English words.
    """
    # TODO: Implement
    # nltk.download('words')
    # words = set(w.lower() for w in nltk.corpus.words.words())
    raise NotImplementedError


def build_ngram_table(benign_domains: list, output_path: str) -> dict:
    """Compute character trigram frequency table from benign domains.

    Returns:
        Dict mapping trigram string -> probability (float).
    """
    # TODO: Implement
    raise NotImplementedError


def run_preprocessing(data_path: str):
    """Full preprocessing pipeline. Called by main.py --mode preprocess."""
    # TODO: Implement
    # 1. load_raw_data()
    # 2. clean_domains()
    # 3. build_english_dictionary()
    # 4. build_ngram_table()
    # 5. train_test_split() with stratify=y, random_state=42
    # 6. Save train.csv, test.csv to data_path
    raise NotImplementedError
