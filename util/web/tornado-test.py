import tornado.web

import tornado.ioloop


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("hello world")


def make_app():
    return tornado.web.Application([
        (r"/", IndexHandler),
    ])


if __name__ == '__main__':
    app = make_app()
    app.listen(address="0.0.0.0", port=8080)
    tornado.ioloop.IOLoop.current().start()
