import requests

post_data = {
    "name": "lisi",
    "pass": "123456"
}
res = requests.post(url="http://localhost:8080/login", data=post_data)
print(res.text)
