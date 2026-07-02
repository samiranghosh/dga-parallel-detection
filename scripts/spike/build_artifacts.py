"""Build the spike's shared artifacts: 5-feature matrices + baseline RF-100.

Outputs (scratchpad ART dir):
  X_train5.npy / y_train.npy / X_test5.npy / y_test.npy
  rf100_5feat.pkl   (baseline config: n_estimators=100, random_state=42)
Prints the test-set accuracy — MUST reproduce 93.179% or the spike stops.
"""
import sys, os, time, pickle, multiprocessing
sys.path.insert(0, r"c:/Users/samir/source/repos/dga-parallel-detection")
import numpy as np
import pandas as pd

ART = r"C:/Users/samir/AppData/Local/Temp/claude/c--Users-samir-source-repos-dga-parallel-detection/932315aa-8a56-47d8-95fa-59c19e89893b/scratchpad/artifacts"

if __name__ == "__main__":
    multiprocessing.freeze_support()
    os.makedirs(ART, exist_ok=True)
    from src.shared_resources import initialize_shared_resources
    from src.parallel_engine import parallel_extract_features
    from src.classifier import train_random_forest

    t0 = time.time()
    dictionary, ngram = initialize_shared_resources("data/")
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    print(f"[art] loaded: train={len(train_df)} test={len(test_df)} [{time.time()-t0:.0f}s]", flush=True)

    t = time.time()
    X_train = parallel_extract_features(train_df["domain"].astype(str).tolist(), 8,
                                        dictionary, ngram, skip_levenshtein=True)
    X_test = parallel_extract_features(test_df["domain"].astype(str).tolist(), 8,
                                       dictionary, ngram, skip_levenshtein=True)
    y_train = train_df["label"].values
    y_test = test_df["label"].values
    np.save(os.path.join(ART, "X_train5.npy"), X_train)
    np.save(os.path.join(ART, "X_test5.npy"), X_test)
    np.save(os.path.join(ART, "y_train.npy"), y_train)
    np.save(os.path.join(ART, "y_test.npy"), y_test)
    print(f"[art] features: X_train={X_train.shape} X_test={X_test.shape} [{time.time()-t:.0f}s]", flush=True)

    t = time.time()
    model = train_random_forest(X_train, y_train)  # defaults: n_estimators=100, rs=42
    acc = (model.predict(X_test) == y_test).mean()
    print(f"[art] RF-100 trained [{time.time()-t:.0f}s]  TEST ACCURACY = {acc*100:.4f}%  "
          f"(baseline claim: 93.1790%)", flush=True)

    p = os.path.join(ART, "rf100_5feat.pkl")
    with open(p, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[art] pickled: {p} ({os.path.getsize(p)/1e6:.1f} MB)", flush=True)
    total_nodes = sum(e.tree_.node_count for e in model.estimators_)
    print(f"[art] total tree nodes: {total_nodes:,}", flush=True)
    print("[art] DONE", flush=True)
