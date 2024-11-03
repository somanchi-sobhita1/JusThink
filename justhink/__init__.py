import os
from venv import logger

def safeGet(key: str):
    value = os.environ.get(key)
    if value is None:
        logger.error(f"ENV NOT FOUND for key: {key}")
        return None
    else:
        return value

def safeGetWithDefault(key: str, default:str):
    value = os.environ.get(key)
    if value is None:
        logger.info(f"ENV NOT FOUND for key: {key}, using default value {default}")
        return default
    else:
        return value

ENV = safeGetWithDefault("ENV", "DEV")
OPENAPI_KEY = safeGet("OPENAPI_KEY")