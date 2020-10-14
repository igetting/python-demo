from flask import Flask
from flask import Response

app = Flask(__name__)


@app.route("/")
def test():
    return Response("hello world")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
