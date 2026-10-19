import json, os, pickle
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_WORDS, OVERLAP = 200, 50