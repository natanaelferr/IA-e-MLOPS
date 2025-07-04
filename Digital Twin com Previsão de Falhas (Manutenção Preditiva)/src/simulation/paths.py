from pathlib import Path

class Paths:
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    DATA_DIR = ROOT_DIR / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROCESSED_DIR = DATA_DIR / "processed"
    MODELS_DIR = ROOT_DIR / "models"

    @classmethod
    def ensure_directories_exist(cls):
        for path in [cls.DATA_DIR, cls.RAW_DIR, cls.PROCESSED_DIR]:
            path.mkdir(parents=True, exist_ok=True)
