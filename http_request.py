import http.client

conn = http.client.HTTPSConnection("http://127.0.0.1:8080/get_random_record")

payload = ""

headers = {}

conn.request("GET", "/get_random_record", payload, headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))