from flask import Flask, request

app = Flask(__name__)


@app.route('/login', methods=['GET', 'POST'])
def hello_world():
    if request.method == 'GET':
        name = request.args['name']
        passwd = request.args['pass']
    elif request.method == 'POST':
        name = request.form['name']
        passwd = request.form['pass']
    print(name, ':', passwd)
    return 'Hello World!'


if __name__ == '__main__':
    app.run(port=8080)
