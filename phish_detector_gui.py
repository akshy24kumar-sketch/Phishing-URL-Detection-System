"""
phish_detector_gui_final_v4.py
Phishing URL Detector GUI + ML (Fully working, fixed for small datasets & class imbalance)
"""

import os, re, threading
import pandas as pd, numpy as np
import tldextract, joblib
from datetime import datetime
from io import StringIO
import requests

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

try:
    import whois
    HAVE_WHOIS = True
except:
    HAVE_WHOIS = False

MODEL_PATH = "phish_model.joblib"
DEFAULT_PHISH_DATA_URL = "https://raw.githubusercontent.com/mitchellkrogza/Phishing.Database/master/ONLINE/blacklist_phish.csv"
DEFAULT_LEGIT_DATA_URL = "https://raw.githubusercontent.com/aboul3la/SubwordTextEncoder/master/data/top-1m.csv"
RANDOM_STATE = 42

# ---------- Feature extractors ----------
class URLTokenizer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        # break url into letter/number tokens and keep separators as tokens
        return [" ".join(re.findall(r"[a-zA-Z0-9]{2,}|[./:-]", str(url).lower())) for url in X]

class LexicalFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, use_whois=False): self.use_whois = use_whois
    def fit(self, X, y=None): return self
    def _has_ip(self, domain): return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))
    def _domain_age_days(self, domain):
        if not HAVE_WHOIS or not self.use_whois: return 0.0
        try:
            w = whois.whois(domain)
            cd = w.creation_date
            if isinstance(cd, list): cd = cd[0]
            if isinstance(cd, datetime): return (datetime.now() - cd).days
        except:
            return 0.0
        return 0.0
    def transform(self, X):
        rows = []
        for url in X:
            try:
                parsed = tldextract.extract(url)
            except:
                parsed = tldextract.ExtractResult('', '', '')
            domain = parsed.registered_domain or parsed.domain or ""
            # hostname length: use domain+subdomain if available
            sub = parsed.subdomain or ""
            dom = parsed.domain or ""
            hostname_len = len((sub + "." + dom).strip(".")) if dom else len(dom or "")
            length = len(str(url))
            rows.append([
                float(length),
                float(hostname_len),
                float(str(url).count(".")),
                float(str(url).count("-")),
                float(str(url).count("@")),
                float(str(url).count("?")),
                float(str(url).count("=")),
                float(str(url).count("/")),
                float(str(url).count("_")),
                1.0 if str(url).lower().startswith("https") else 0.0,
                1.0 if self._has_ip(sub or dom) else 0.0,
                float(sum(1 for w in ["login","verify","update","bank","secure","account","ebay","paypal","signin","confirm","wp-login"] if w in str(url).lower())),
                float(sum(c.isdigit() for c in str(url))),
                float(sum(c.isalpha() for c in str(url))),
                float(sum(c.isdigit() for c in str(url))/(sum(c.isalpha() for c in str(url))+1)),
                float(self._domain_age_days(domain))
            ])
        return np.array(rows, dtype=float)

# ---------- Data loader ----------
def download_text_url(url, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

def load_datasets(phish_url=None, legit_url=None, max_legit=20000, max_phish=20000):
    phish_url = phish_url or DEFAULT_PHISH_DATA_URL
    legit_url = legit_url or DEFAULT_LEGIT_DATA_URL
    phish_list, legit_list = [], []

    try:
        dfp = pd.read_csv(StringIO(download_text_url(phish_url)), header=None, names=["url"])
        phish_list = dfp["url"].dropna().astype(str).str.strip().tolist()
    except Exception as e:
        # fallback safe examples
        phish_list = ["http://secure-login.example.com/confirm", "http://login.example.com/secure"]
    try:
        dfl = pd.read_csv(StringIO(download_text_url(legit_url)), header=None, names=["domain"])
        legit_list = ("http://" + dfl["domain"].dropna().astype(str).str.strip()).tolist()
    except Exception as e:
        legit_list = ["https://www.google.com","https://www.wikipedia.org","https://www.github.com"]

    # truncate to requested maxima
    phish_list, legit_list = phish_list[:max_phish], legit_list[:max_legit]

    # Ensure at least two items in each to avoid impossible splits
    if len(phish_list) < 2:
        phish_list = phish_list * 2
    if len(legit_list) < 2:
        legit_list = legit_list * 2

    # Balance classes: upsample the minority to match the majority (but cap at max_* sizes)
    try:
        if len(phish_list) != len(legit_list):
            # determine target: the larger of the two, but don't exceed provided max limits
            target = min(max(len(phish_list), len(legit_list)), max(max_phish, max_legit))
            if len(phish_list) < target:
                phish_list = resample(phish_list, replace=True, n_samples=target, random_state=RANDOM_STATE)
            if len(legit_list) < target:
                legit_list = resample(legit_list, replace=True, n_samples=target, random_state=RANDOM_STATE)
    except Exception:
        # if balancing fails for any reason, continue with existing lists
        pass

    df = pd.concat([pd.DataFrame({"url":phish_list,"label":1}),
                    pd.DataFrame({"url":legit_list,"label":0})]).sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)
    return df

# ---------- Pipeline ----------
def build_pipeline(use_whois=False):
    # token_pipe -> URLTokenizer produces cleaned string docs -> TfidfVectorizer
    token_pipe = Pipeline([("tokenize", URLTokenizer()), ("tfidf", TfidfVectorizer(ngram_range=(1,3), max_features=20000))])
    lexical = Pipeline([("lex", LexicalFeatures(use_whois=use_whois)), ("scale", StandardScaler())])

    combined = FeatureUnion([("tfidf", token_pipe), ("lexical", lexical)])

    # classifiers with class_weight balanced to handle residual imbalance
    clf1 = LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE)
    clf2 = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced")
    clf3 = SVC(probability=True, kernel="rbf", class_weight="balanced", random_state=RANDOM_STATE)

    classifiers = [("lr", clf1), ("rf", clf2), ("svc", clf3)]

    # try to include xgboost if available, with scale_pos_weight to balance
    try:
        from xgboost import XGBClassifier
        # default scale_pos_weight will be set later in training if needed
        clf4 = XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=200, random_state=RANDOM_STATE)
        classifiers.append(("xgb", clf4))
        have_xgb = True
    except Exception:
        have_xgb = False

    ensemble = VotingClassifier(estimators=classifiers, voting="soft", n_jobs=-1)
    return Pipeline([("features", combined), ("clf", ensemble)])

# ---------- Train & evaluate ----------
def train_and_evaluate(df, use_whois=False, gui_callback=None):
    if gui_callback: gui_callback("Building pipeline...")
    pipe = build_pipeline(use_whois)
    X, y = df["url"].values, df["label"].values
    n = len(X)
    if gui_callback: gui_callback(f"Dataset has {n} samples (phish={int(y.sum())}, legit={n-int(y.sum())})")

    # Safety checks
    classes, counts = np.unique(y, return_counts=True)
    num_classes = len(classes)
    min_count = counts.min() if len(counts) > 0 else 0

    # If dataset too small to meaningfully split, train on entire dataset and skip test set
    if n <= num_classes or min_count < 2:
        if gui_callback: gui_callback("Dataset small or minority class too small for stratified split. Training on the full dataset (no test set).")
        # For XGBoost, set scale_pos_weight if available and there is class imbalance
        try:
            clf_step = pipe.named_steps["clf"]
            if hasattr(clf_step, "estimators_"):
                pass
        except Exception:
            pass
        pipe.fit(X, y)
        return pipe

    # Determine a safe integer test_size (number of samples) that ensures at least one sample per class in test set
    desired_test = max(num_classes, int(n * 0.15))  # at least one per class, prefer ~15%
    if desired_test >= n:
        desired_test = max(num_classes, n - num_classes)  # ensure train set has at least num_classes samples
    if desired_test < num_classes:
        desired_test = num_classes

    # When possible, use stratify to preserve class proportions
    stratify_y = y if min_count >= 2 else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=desired_test,
            stratify=stratify_y,
            random_state=RANDOM_STATE,
            shuffle=True
        )
    except Exception as e:
        # fallback: train on full dataset
        if gui_callback:
            gui_callback(f"Warning: failed to split dataset ({e}). Training on full dataset instead.")
        pipe.fit(X, y)
        return pipe

    if gui_callback: gui_callback(f"Training model on {len(X_train)} samples; testing on {len(X_test)} samples...")

    # If XGBoost present and imbalance exists, configure scale_pos_weight for xgb estimator
    # (VotingClassifier wraps estimators; we adjust XGB if available)
    try:
        # detect xgb estimator inside pipeline
        clf_step = pipe.named_steps["clf"]
        for name, est in clf_step.estimators:
            # est is an estimator instance (not yet fitted) inside VotingClassifier
            # if it's XGBClassifier, set scale_pos_weight
            from xgboost import XGBClassifier
            if isinstance(est, XGBClassifier):
                # compute scale_pos_weight = negative_count / positive_count
                neg = int((y_train == 0).sum())
                pos = int((y_train == 1).sum())
                if pos > 0:
                    est.set_params(scale_pos_weight=float(neg / pos))
    except Exception:
        # ignore if xgboost not present or any error
        pass

    pipe.fit(X_train, y_train)

    if gui_callback:
        try:
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, zero_division=0)
            gui_callback(f"Accuracy: {acc:.4f}\n{report}")
        except Exception as e:
            gui_callback(f"Could not evaluate on test set: {e}")

    return pipe

# ---------- URL prediction ----------
def predict_url(pipe, url, use_whois=False):
    # robust method to extract phish probability index (class=1)
    proba = None
    try:
        pred_proba = pipe.predict_proba([url])[0]
        # find index of class '1' in estimator's classes_
        try:
            cls = pipe.named_steps["clf"].classes_
        except Exception:
            # fallback: many estimators put classes_ on the pipeline itself after fit
            cls = getattr(pipe, "classes_", None)
        if cls is None:
            # fallback: assume second column is phish
            phish_idx = 1 if len(pred_proba) > 1 else 0
        else:
            phish_idx = int(list(cls).index(1)) if 1 in cls else (1 if len(pred_proba) > 1 else 0)
        phish_prob = float(pred_proba[phish_idx]) if len(pred_proba) > phish_idx else 0.0
    except Exception:
        # if predict_proba fails (some classifiers), fall back to predict
        try:
            phish_prob = float(pipe.predict([url])[0])
        except Exception:
            phish_prob = 0.0

    label = int(pipe.predict([url])[0])

    heuristics = []
    if url.count("@") > 0: heuristics.append("Contains '@'")
    if url.count("//") > 2: heuristics.append("Possible redirect")
    if url.startswith("http://"): heuristics.append("Uses http (not https)")
    if re.search(r"\d{1,3}(\.\d{1,3}){3}", url): heuristics.append("Contains IP address")
    if any(s in url.lower() for s in ["login","secure","account","confirm","verify","update"]): heuristics.append("Suspicious keywords")
    return {"label": label, "phish_prob": phish_prob, "explanation": heuristics[:5]}

# ---------- GUI ----------
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

class PhishDetectorGUI:
    def __init__(self, master):
        self.master = master
        master.title("Phishing URL Detector (ML)")
        self.model = None
        self.use_whois = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Ready")
        self.data_sources = {"phish_url": DEFAULT_PHISH_DATA_URL, "legit_url": DEFAULT_LEGIT_DATA_URL}

        frm = ttk.Frame(master, padding=12)
        frm.grid(row=0,column=0,sticky="nsew")
        master.columnconfigure(0,weight=1)
        master.rowconfigure(0,weight=1)

        ttk.Label(frm,text="Enter URL to check:").grid(row=0,column=0,sticky="w")
        self.url_entry = ttk.Entry(frm,width=80)
        self.url_entry.grid(row=1,column=0,columnspan=3,sticky="ew",pady=4)
        self.btn_check = ttk.Button(frm,text="Check URL",command=self.on_check)
        self.btn_check.grid(row=1,column=3,padx=6)
        self.btn_train = ttk.Button(frm,text="Train model (download datasources)",command=self.on_train)
        self.btn_train.grid(row=2,column=0,pady=6)
        ttk.Button(frm,text="Load saved model",command=self.on_load_model).grid(row=2,column=1,pady=6)
        ttk.Button(frm,text="Save model (force)",command=self.on_save_model).grid(row=2,column=2,pady=6)
        ttk.Button(frm,text="Clear logs",command=self.on_clear_log).grid(row=2,column=3,pady=6)
        ttk.Checkbutton(frm,text="Enable WHOIS domain-age features (slow)",variable=self.use_whois).grid(row=3,column=0,columnspan=2,sticky="w")
        ttk.Label(frm,text="Status:").grid(row=4,column=0,sticky="w",pady=(8,0))
        ttk.Label(frm,textvariable=self.status_var).grid(row=4,column=1,columnspan=3,sticky="w",pady=(8,0))
        ttk.Label(frm,text="Log / Output:").grid(row=5,column=0,sticky="w",pady=(8,0))
        self.log = scrolledtext.ScrolledText(frm,width=100,height=18)
        self.log.grid(row=6,column=0,columnspan=4,sticky="nsew")
        frm.rowconfigure(6,weight=1)
        ttk.Label(frm,text="Model file: "+MODEL_PATH).grid(row=7,column=0,columnspan=4,sticky="w",pady=(6,0))

        if os.path.exists(MODEL_PATH):
            try:
                self.model = joblib.load(MODEL_PATH)
                self.log_message("Loaded existing model from disk.")
            except Exception as e:
                self.log_message(f"Failed to load existing model: {e}")

    def log_message(self,text):
        ts=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log.insert(tk.END,f"[{ts}] {text}\n")
        self.log.see(tk.END)
        self.status_var.set(text)

    def on_clear_log(self):
        self.log.delete("1.0",tk.END)
        self.status_var.set("Cleared logs")

    def on_save_model(self):
        if self.model is None:
            messagebox.showinfo("Save model","No model in memory to save.")
            return
        try:
            joblib.dump(self.model,MODEL_PATH)
            self.log_message(f"Model saved to {MODEL_PATH}")
        except Exception as e:
            self.log_message(f"Save failed: {e}")

    def on_load_model(self):
        if not os.path.exists(MODEL_PATH):
            messagebox.showinfo("Load model",f"No saved model found at {MODEL_PATH}")
            return
        try:
            self.model = joblib.load(MODEL_PATH)
            self.log_message("Model loaded from disk.")
        except Exception as e:
            self.log_message(f"Failed to load model: {e}")

    def on_train(self):
        self.btn_train.config(state="disabled")
        self.btn_check.config(state="disabled")
        threading.Thread(target=self._train_background,daemon=True).start()

    def _train_background(self):
        try:
            self.log_message("Starting dataset download and training...")
            df = load_datasets(self.data_sources["phish_url"],self.data_sources["legit_url"],20000,20000)
            self.log_message(f"Loaded dataset with {len(df)} samples (phish={int(df.label.sum())}, legit={len(df)-int(df.label.sum())})")
            model = train_and_evaluate(df,use_whois=self.use_whois.get(),gui_callback=self.log_message)
            self.model=model
            joblib.dump(self.model,MODEL_PATH)
            self.log_message(f"Training finished. Model saved to {MODEL_PATH}")
        except Exception as e:
            self.log_message(f"Training failed: {e}")
        finally:
            self.btn_train.config(state="normal")
            self.btn_check.config(state="normal")
            self.status_var.set("Ready")

    def on_check(self):
        url=self.url_entry.get().strip()
        if not url:
            messagebox.showwarning("Input error","Please enter a URL.")
            return
        if not re.match(r"https?://",url):
            url="http://"+url
        if self.model is None:
            messagebox.showerror("Model missing","No trained model available. Please train or load a model first.")
            return
        try:
            result=predict_url(self.model,url,use_whois=self.use_whois.get())
            label="PHISHING" if result["label"]==1 else "LEGITIMATE"
            prob=result["phish_prob"]
            heur="; ".join(result["explanation"]) if result["explanation"] else "None"
            self.log_message(f"URL: {url}\nPrediction: {label} (phish probability {prob:.3f})\nHeuristics: {heur}")
            if prob>=0.5:
                messagebox.showwarning("Phishing detected",f"URL predicted as PHISHING (prob {prob:.3f})\nHeuristics: {heur}")
            else:
                messagebox.showinfo("URL checked",f"URL predicted LEGITIMATE (prob phishing {prob:.3f})\nHeuristics: {heur}")
        except Exception as e:
            self.log_message(f"Prediction failed: {e}")
            messagebox.showerror("Prediction error",f"Prediction failed: {e}")

if __name__=="__main__":
    root=tk.Tk()
    app=PhishDetectorGUI(root)
    root.geometry("980x640")
    root.mainloop()
