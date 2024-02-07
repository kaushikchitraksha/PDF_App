from chromadb.config import Settings

CHROMA_SETTINGS = Settings(
    persist_directory='db',
    anonymized_telemetry=False
)
