import requests
import json

def call_external_api(id, embedding):
    # API的URL
    url = 'http://103.39.218.177:8088/api/v1/governance/save-data'

    # 设置请求头
    headers = {
        'Content-Type': 'application/json'
    }

    # 将一维列表转换为JSON字符串
    embedding_str = json.dumps(embedding)

    # 设置请求体
    data = {
        'key': id,
        'value': embedding_str
    }

    # 发送POST请求
    response = requests.post(url, json=data, headers=headers)

    return response
