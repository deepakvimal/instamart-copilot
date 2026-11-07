import glob
import os
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mock_llm                                          # lives in the repo root

POLICY_DIR = os.path.join("data", "policies")


def chunk_words(words, size=40, overlap=10):
    """Chunks of `size` words, each starting `size - overlap` words after the last one."""
    step = size - overlap
    chunks, start = [], 0
    while True:
        chunks.append(" ".join(words[start:start + size]))
        if start + size >= len(words):          # this chunk reaches the last word: stop
            break
        start += step
    return chunks


class PolicyIndex:
    def __init__(self, size=40, overlap=10, folder=POLICY_DIR):
        self.chunks = []
        for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
            with open(path, encoding="utf-8") as f:
                words = f.read().split()
            for text in chunk_words(words, size, overlap):
                self.chunks.append({"source": os.path.basename(path), "text": text})
        self.vectorizer = TfidfVectorizer()
        self.matrix = self.vectorizer.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix)[0]
        order = (-sims).argsort(kind="stable")[:k]       # stable sort: ties resolve the same way every run
        return [{"source": self.chunks[i]["source"],
                 "score": round(float(sims[i]), 4),
                 "text": self.chunks[i]["text"]} for i in order]


_INDEX = PolicyIndex()                                   # chunks of 40 words, overlap 10


def retrieve(query, k=3):
    return _INDEX.retrieve(query, k)


SYSTEM = ("You are a POLICY ASSISTANT for Instamart support. Answer only from the supplied "
          "context. If the context does not contain the answer, say that you could not find "
          "the information.")


def answer(question, index=None):
    index = _INDEX if index is None else index
    hits = index.retrieve(question, 3)
    context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
    user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
    resp = mock_llm.chat([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}])
    return resp.choices[0].message.content, [h["source"] for h in hits]