import json
import hashlib

def generate_hash_uid(to_hash: dict | tuple | list | str) -> str:
    """Generates a unique hash for a given model and arguments."""
    json_string = json.dumps(to_hash, sort_keys=True)

    hash_object = hashlib.sha256(json_string.encode())
    hash_uid = hash_object.hexdigest()

    return hash_uid