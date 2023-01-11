import flask
from data_processing import Data
import tensorflow as tf
import numpy as np

app = flask.Flask(__name__, template_folder='./templates')

print('Loading models...')
# diameter = tf.keras.models.load_model('../Plant_variable_predictor/checkpoints/rgb_diameter')
# height = tf.keras.models.load_model('../Plant_variable_predictor/checkpoints/depth_height')
# leaf_area = tf.keras.models.load_model('../Plant_variable_predictor/checkpoints/rgb_augm_leafarea')
# fresh_weight = tf.keras.models.load_model('../Plant_variable_predictor/checkpoints/rgb-d_freshweight')
# dry_weight = tf.keras.models.load_model('../Plant_variable_predictor/checkpoints/rgb_dryweight')
print('Models loaded.')


@app.route('/', methods=['GET', 'POST'])
def index():
    return flask.render_template('home.html')

@app.route('/image', methods=['POST'])
def predict_image():
    rgb = flask.request.files.get('rgb', '')
    depth = flask.request.files.get('depth', '')
    
    data = Data()
    
    rgb = data.process_img(rgb)
    depth = data.process_img(depth)

    plant_values = {'diameter': diameter.predict(rgb),
                    'height': height.predict(depth),
                    'leaf_area': leaf_area.predict(rgb),
                    'freshweight': fresh_weight.predict(np.dstack((rgb.numpy(), depth.numpy()))),
                    'dryweight': dry_weight.predict(rgb)}
    return flask.render_template('extraction.html', result_plant=plant_values)


@app.route('/harvest', methods=['POST'])
def predict_harvest():
    form_values = (
        flask.request.form.get('species'), flask.request.form.get('height'), flask.request.form.get('diameter'),
        flask.request.form.get('leafarea'), flask.request.form.get('freshweight'), flask.request.form.get('dryweight'))

    processed = Data.process_plant_values(form_values)

    # TODO load model, make pred
    predicted_date = '11/09/2001'
    return flask.render_template('result.html', result_time=predicted_date)

if __name__ == '__main__':
    app.run()
