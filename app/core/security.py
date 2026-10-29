import hashlib

def get_file_hash(file_bytes: bytes) -> str:
    """Computes SHA-256 hash of file bytes for uniqueness and integrity verification."""
    return hashlib.sha256(file_bytes).hexdigest()
