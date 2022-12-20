import flask

app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    message = ''
    if flask.request.method == 'POST':
        message = 'Hello ' + flask.request.form['name-input'] + '!'
    return flask.render_template('index.html', message=message)


if __name__ == '__main__':
    app.run()
