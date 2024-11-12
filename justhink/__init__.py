import base64
import os
import boto3
from venv import logger


def safeGetWithDefault(key: str, default:str):
    value = os.environ.get(key)
    if value is None:
        logger.info(f"ENV NOT FOUND for key: {key}, using default value {default}")
        return default
    else:
        return value
    
def safeGet(key: str):
    value = os.environ.get(key)
    if value is None:
        logger.error(f"ENV NOT FOUND for key: {key}")
        return None
    else:
        return value
    
AWS_REGION=safeGetWithDefault("AWS_REGION", "ap-south-1")
def getKmsDecryptedValue(value: str):
    try:
    # kms_client = boto3.client('kms', region_name=AWS_REGION, endpoint_url='http://localhost:4566')
        kms_client = boto3.client('kms', region_name=AWS_REGION)
        encrypted_key = base64.b64decode(value)
        response = kms_client.decrypt(
            CiphertextBlob=encrypted_key
        )
        return response['Plaintext'].decode('utf-8')
    except:
        return safeGet("OPENAPI_KEY")

ENV = safeGetWithDefault("ENV", "DEV")
OPENAPI_KEY = OPENAPI_KEY = getKmsDecryptedValue(safeGet("OPENAPI_KEY"))