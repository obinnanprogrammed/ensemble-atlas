from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    data_dir: Path = Path("data")
    model_id: str = "laion/clap-htsat-unfused"
    model_revision: str = "8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a"
    preprocessing_version: str = "mono-polyphase-48k-window10-stride5-zero-pad-v1"
    dimension: int = 512
    lease_seconds: float = 120.0
    heartbeat_seconds: float = 20.0
    max_attempts: int = 3
    cpu_threads: int = 2

    def __post_init__(self):
        object.__setattr__(self, "data_dir", self.data_dir.expanduser().resolve())
        if not 0 < self.heartbeat_seconds < self.lease_seconds:
            raise ValueError("Heartbeat interval must be positive and shorter than the lease.")
        if self.max_attempts < 1:
            raise ValueError("Attempt limit must be at least one.")
        if self.cpu_threads < 1:
            raise ValueError("CPU thread count must be at least one.")

    @property
    def contract(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "preprocessing_version": self.preprocessing_version,
            "dimension": self.dimension,
        }


SAMPLE_RATE = 48_000
WINDOW_SECONDS = 10
STRIDE_SECONDS = 5
