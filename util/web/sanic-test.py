from sanic import Sanic
from sanic.response import HTTPResponse

app = Sanic("test")


@app.route("/")
def test(request):
    return HTTPResponse("hello world")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080)
