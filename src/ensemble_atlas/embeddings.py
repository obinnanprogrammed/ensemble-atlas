from typing import Protocol

import numpy as np

from .config import Config, SAMPLE_RATE


def normalize(vector, dimension: int) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32)
    if result.shape != (dimension,) or not np.isfinite(result).all():
        raise ValueError(f"Expected a finite embedding of dimension {dimension}.")
    norm = float(np.linalg.norm(result))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Embedding must have a finite, nonzero norm.")
    return result / norm


class Embedder(Protocol):
    def audio(self, samples: np.ndarray) -> np.ndarray: ...
    def text(self, query: str) -> np.ndarray: ...


class ClapEmbedder:
    def __init__(self, config: Config):
        # Heavy imports stay out of ingest/status and fake-embedding tests.
        import torch
        from transformers import ClapModel, ClapProcessor

        self.config = config
        self.torch = torch
        torch.set_num_threads(config.cpu_threads)
        options = {"revision": config.model_revision, "cache_dir": str(config.data_dir / "models")}
        self.processor = ClapProcessor.from_pretrained(config.model_id, **options)
        self.model = ClapModel.from_pretrained(config.model_id, **options).to("cpu").eval()
        if self.model.config.projection_dim != config.dimension:
            raise ValueError("CLAP projection dimension does not match this index.")

    def audio(self, samples: np.ndarray) -> np.ndarray:
        inputs = self.processor(audios=samples, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with self.torch.inference_mode():
            features = self.model.get_audio_features(**inputs)
        return normalize(features[0].cpu().numpy(), self.config.dimension)

    def text(self, query: str) -> np.ndarray:
        inputs = self.processor(text=[query], padding=True, truncation=True, return_tensors="pt")
        with self.torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return normalize(features[0].cpu().numpy(), self.config.dimension)
