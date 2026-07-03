"""
Granite ONNX Embedder — Wrapper ONNX int8 para sentence-transformers compat.

Carrega granite-embedding-97m-multilingual-r2 via ONNX Runtime (int8 quantizado).
Fornece interface compatível com SentenceTransformer.encode().
"""
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_DIR = os.path.expanduser("~/.hermes/models/granite-embedding-97m-multilingual-r2")
ONNX_PATH = os.path.join(MODEL_DIR, "onnx", "model_quint8_avx2.onnx")
EMBEDDING_DIM = 384
MAX_LENGTH = 8192


class GraniteONNXEmbedder:
    """Embedder usando granite-97m ONNX int8 com mean pooling + L2 normalize."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        self.session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
        self.input_names = [inp.name for inp in self.session.get_inputs()]

    def encode(self, texts, normalize_embeddings=True):
        """Codifica lista de textos para embeddings vetoriais."""
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=MAX_LENGTH
        )

        ort_inputs = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"]
        }
        if "token_type_ids" in self.input_names:
            ort_inputs["token_type_ids"] = inputs.get(
                "token_type_ids",
                np.zeros_like(inputs["input_ids"])
            )

        outputs = self.session.run(None, ort_inputs)
        token_embeddings = outputs[0]
        attention_mask = inputs["attention_mask"]

        # Mean pooling
        mask = np.expand_dims(attention_mask, axis=-1).astype(token_embeddings.dtype)
        sum_embeddings = np.sum(token_embeddings * mask, axis=1)
        sum_mask = np.clip(np.sum(mask, axis=1), a_min=1e-9, a_max=None)
        pooled = sum_embeddings / sum_mask

        # L2 normalize (se solicitado)
        if normalize_embeddings:
            norm = np.linalg.norm(pooled, axis=1, keepdims=True)
            pooled = pooled / norm

        return pooled.tolist()
