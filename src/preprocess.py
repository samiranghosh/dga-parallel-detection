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
import gzip
import json
import logging
import pandas as pd
import numpy as np
import nltk
import pickle
import tldextract
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def load_raw_data(data_path: str) -> pd.DataFrame:
    """Load and combine DGA + benign domain JSONL.gz into a single DataFrame.

    Returns:
        DataFrame with columns: ['domain', 'label']
        label: 1 = DGA (malicious), 0 = Normal (benign)
    """
    file_path = os.path.join(data_path, 'dga-training-data-encoded.json.gz')
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}. Did you clone with Git LFS?")

    logger.info(f"Loading raw data from {file_path}...")
    
    benign_records = []
    dga_records = []

    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            try:
                data = json.loads(line)
                domain = data.get('domain')
                threat = data.get('threat')
                
                if not domain or not threat:
                    continue
                
                label = 1 if threat == 'dga' else 0
                
                if label == 1 and len(dga_records) < 500000:
                    dga_records.append({'domain': domain, 'label': label})
                elif label == 0 and len(benign_records) < 500000:
                    benign_records.append({'domain': domain, 'label': label})
                
                if len(dga_records) == 500000 and len(benign_records) == 500000:
                    break
            except json.JSONDecodeError:
                continue

    df = pd.DataFrame(benign_records + dga_records)
    logger.info(f"Loaded {len(df)} overall domains (DGA: {len(dga_records)}, Benign: {len(benign_records)})")
    return df


def strip_tld(domain: str) -> str:
    """Remove TLD suffix from domain using tldextract.

    Example: 'evil-domain.co.uk' -> 'evil-domain'
    """
    ext = tldextract.extract(domain)
    # Reconstruct without the suffix (TLD). If there's a subdomain, include it since DGA can be there
    if ext.subdomain:
        return f"{ext.subdomain}.{ext.domain}"
    return ext.domain


def clean_domains(df: pd.DataFrame) -> pd.DataFrame:
    """Clean domain DataFrame: lowercase, strip TLD, deduplicate, sort.

    Returns:
        Cleaned DataFrame sorted alphabetically by domain.
    """
    logger.info("Cleaning domains...")
    # Lowercase
    df['domain'] = df['domain'].str.lower()
    
    # Strip TLD
    df['domain'] = df['domain'].apply(strip_tld)
    
    # Drop rows where domain became empty
    df = df[df['domain'] != '']
    
    # Deduplicate
    initial_len = len(df)
    df = df.drop_duplicates(subset=['domain'])
    logger.info(f"Deduplicated {initial_len - len(df)} rows.")
    
    # Sort alphabetically (critical for Levenshtein distance feature)
    df = df.sort_values(by='domain').reset_index(drop=True)
    return df


def build_english_dictionary(output_path: str) -> set:
    """Download NLTK words corpus and save as a text file.

    Returns:
        Set of lowercase English words.
    """
    logger.info("Building English dictionary...")
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')
        
    words = set(w.lower() for w in nltk.corpus.words.words())
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for word in sorted(words):
            f.write(word + '\n')
            
    logger.info(f"Saved {len(words)} words to {output_path}")
    return words


def build_ngram_table(benign_domains: list, output_path: str) -> dict:
    """Compute character trigram frequency table from benign domains.

    Returns:
        Dict mapping trigram string -> probability (float).
    """
    logger.info("Building trigram table from benign domains...")
    trigram_counts = {}
    total_trigrams = 0
    
    for domain in benign_domains:
        domain = str(domain)
        # Pad strings to handle start/end trigrams (optional, but standard for NGrams)
        # We will just extract raw trigrams here as per standard pronounceability score
        for i in range(len(domain) - 2):
            trigram = domain[i:i+3]
            trigram_counts[trigram] = trigram_counts.get(trigram, 0) + 1
            total_trigrams += 1
            
    trigram_probs = {k: v / total_trigrams for k, v in trigram_counts.items()}
    
    with open(output_path, 'wb') as f:
        pickle.dump(trigram_probs, f)
        
    logger.info(f"Saved {len(trigram_probs)} unique trigrams to {output_path}")
    return trigram_probs


def run_preprocessing(data_path: str):
    """Full preprocessing pipeline. Called by main.py --mode preprocess."""
    logger.info("Starting preprocessing pipeline...")
    
    os.makedirs(data_path, exist_ok=True)
    
    df = load_raw_data(data_path)
    df = clean_domains(df)
    
    build_english_dictionary(os.path.join(data_path, 'english_dictionary.txt'))
    
    benign_domains = df[df['label'] == 0]['domain'].tolist()
    build_ngram_table(benign_domains, os.path.join(data_path, 'ngram_table.pkl'))
    
    logger.info("Splitting dataset 80/20...")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    train_path = os.path.join(data_path, 'train.csv')
    test_path = os.path.join(data_path, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Saved train set ({len(train_df)} rows) to {train_path}")
    logger.info(f"Saved test set ({len(test_df)} rows) to {test_path}")
    logger.info("Preprocessing complete.")
