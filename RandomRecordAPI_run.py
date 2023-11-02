import requests

# Assuming the Flask API is available at 127.0.0.1:8080
api_url = "http://127.0.0.1:8080/get_random_record"
response = requests.get(api_url)

if response.status_code == 200:
    # Check if the response contains a DataFrame as HTML
    if "table" in response.text:
        from IPython.display import HTML, display
        display(HTML(response.text))
    else:
        print("API request was successful, but the response format is unexpected.")
else:
    print(f"API request failed with status code: {response.status_code}")
