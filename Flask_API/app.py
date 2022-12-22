import flask
from data_processing import Data

app = flask.Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    return flask.render_template('index.html')


@app.route('/image', methods=['POST'])
def predict_image():
    value = flask.request.files.get('image_of_lettuce', '')
    img = Data.process_img(value)

    # TODO load model, make pred
    plant_values = []
    return flask.render_template('index.html', result_plant=plant_values)


@app.route('/harvest', methods=['POST'])
def predict_harvest():
    form_values = (flask.request.form.get('species'), flask.request.form.get('height'), flask.request.form.get('diameter'),
                   flask.request.form.get('leafarea'), flask.request.form.get('freshweight'), flask.request.form.get('dryweight'))

    processed = Data.process_plant_values(form_values)

    # TODO load model, make pred
    predicted_date = '11/09/2001'
    return flask.render_template('index.html', result_time=predicted_date)





if __name__ == '__main__':
    app.run()
