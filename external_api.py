import requests
import json

def call_external_api(id, embedding):
    """
    Calls an external API to save face embedding data to blockchain.
    Parameters:
        id (str): The unique identifier for the subject.
        embedding (list): The face embedding data.
    Returns:
        Response: The response object from the API request.
    """

    # API endpoint URL
    url = 'http://103.39.218.177:8088/api/v1/governance/save-data'
    
    # Set the headers to indicate JSON content type
    headers = {
        'Content-Type': 'application/json'
    }

    # Convert the embedding to a JSON string
    embedding_str = json.dumps(embedding)

    # Prepare the data payload for the POST request
    data = {
        'key': id,
        'value': embedding_str
    }

    # Make a POST request to the external API
    try:
        response = requests.post(url, json=data, headers=headers)
        return response
    except requests.exceptions.RequestException as e:
        # Handle any exceptions that occur during the request
        print(f"An error occurred: {e}")
        return None
