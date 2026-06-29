from sklearn.base import ClassifierMixin, BaseEstimator
import torch
import numpy as np

class HarmDetector(BaseEstimator, ClassifierMixin):
    """
    Assumes input df has a 'text' column for the input text.
    """

    def __init__(self, bert_model=None, tokenizer=None, device=None, classification_head='linear_svc'):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.classification_head = classification_head
        if classification_head == 'linear_svc':
            from sklearn.svm import LinearSVC
            self.classifier = LinearSVC(random_state=42, max_iter=10000)
        elif classification_head == 'logistic_regression':
            from sklearn.linear_model import LogisticRegression
            self.classifier = LogisticRegression(random_state=42, max_iter=10000)
        else:
            raise ValueError(f"Unknown classification head: {classification_head}")

    def get_cls_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # CLS token is the first token ([CLS])
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_emb

    def get_cls_embedding_batch(self, texts):
        if not texts:
            return np.empty((0, 0))
        inputs = self.tokenizer(
            list(texts),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return cls_emb

    def fit(self, X, y):
        X_embedded = self.get_cls_embedding_batch(X['text'].tolist())
        self.classifier.fit(X_embedded, y)
        return self

    def predict(self, X):
        X_embedded = self.get_cls_embedding_batch(X['text'].tolist())
        return self.classifier.predict(X_embedded)
    
    def predict_proba(self, X):
        X_embedded = self.get_cls_embedding_batch(X['text'].tolist())
        probs = self.classifier.predict_proba(X_embedded)
        return probs
    
    def get_params(self, deep=True):
        return {
            "bert_model": self.bert_model,
            "tokenizer": self.tokenizer,
            "device": self.device,
            "classification_head": self.classification_head
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self