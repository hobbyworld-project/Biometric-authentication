import requests
import json

def call_external_api(id, embedding):
    # API的URL
    url = 'http://103.39.218.177:8088/api/v1/governance/save-data'
    
    headers = {
        'Content-Type': 'application/json'
    }

    embedding_str = json.dumps(embedding)

    data = {
        'key': id,
        'value': embedding_str
    }

    response = requests.post(url, json=data, headers=headers)

    return response
