from sklearn.base import ClassifierMixin, BaseEstimator
import torch
import numpy as np
from sklearn.svm import LinearSVC

class HarmDetector(BaseEstimator, ClassifierMixin):
    """
    Assumes input df has a 'text' column for the input text.
    """

    def __init__(self, bert_model=None, tokenizer=None, device=None):
        self.bert_model = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.classifier = LinearSVC(random_state=42, max_iter=10000)


    def get_cls_embedding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # CLS token is the first token ([CLS])
            cls_emb = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return cls_emb

    def fit(self, X, y):
        enmbeddings = X['text'].apply(self.get_cls_embedding)
        X_embedded = np.vstack(enmbeddings)
        self.classifier.fit(X_embedded, y)
        return self

    def predict(self, X):
        embeddings = X['text'].apply(self.get_cls_embedding)
        X_embedded = np.vstack(embeddings)
        return self.classifier.predict(X_embedded)
    
    def get_params(self, deep=True):
        return {
            "bert_model": self.bert_model,
            "tokenizer": self.tokenizer,
            "device": self.device
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self