import pickle
import flask
import pandas as pd

from data_processing import Data
import tensorflow as tf
import numpy as np

app = flask.Flask(__name__, template_folder='./templates')

print('Loading models...')

try:
    diameter = tf.keras.models.load_model(r"../Plant_variable_predictor/checkpoints/rgb_augm_reg_diameter/cp-0029.ckpt")
    height = tf.keras.models.load_model(r"../Plant_variable_predictor/checkpoints/depth_reg_height/cp-0098.ckpt")
    leaf_area = tf.keras.models.load_model("../Plant_variable_predictor/checkpoints/rgb_augm_reg_leafarea/cp-0035.ckpt")
    fresh_weight = tf.keras.models.load_model("../Plant_variable_predictor/checkpoints/rgb-d_freshweight/cp-0100.ckpt")
    dry_weight = tf.keras.models.load_model("../Plant_variable_predictor/checkpoints/rgb_dryweight/cp-0077.ckpt")
    harvest = pickle.load(open('../Forecasting/linear_model-all-varieties.sav', 'rb'))
    harvest_wo_variety = pickle.load(open('../Forecasting/linear_model-all-varieties-no-dum.sav', 'rb'))
except:
    print("Could not load models. Please check if the models are in the right directories.")

print('Models loaded.')

data = Data()


@app.route('/', methods=['GET', 'POST'])
def index():
    return flask.render_template('home.html')


@app.route('/image', methods=['POST'])
def predict_image():
    rgb_img_req = flask.request.files.get("rgb")
    depth_img_req = flask.request.files.get("depth")

    assert rgb_img_req is not None
    assert depth_img_req is not None

    rgb_bytes_img = rgb_img_req.read()
    depth_bytes_img = depth_img_req.read()

    normalized_images = data.prepare_images(image=rgb_bytes_img,
                                            image_depth=depth_bytes_img,
                                            requires_normalization=True)

    not_normalized_images = data.prepare_images(image=rgb_bytes_img,
                                                image_depth=depth_bytes_img,
                                                requires_normalization=False)

    plant_values = {'diameter': diameter.predict(normalized_images['rgb'])[0][0],
                    'height': height.predict(normalized_images['depth'])[0][0],
                    'leaf_area': leaf_area.predict(normalized_images['rgb'])[0][0],
                    'freshweight': fresh_weight.predict(
                        np.concatenate((not_normalized_images['rgb'], not_normalized_images['depth']), axis=-1))[0][0],
                    'dryweight': dry_weight.predict(not_normalized_images['rgb'])[0][0]}

    print(plant_values['diameter'])
    return flask.render_template('extraction.html', result_plant=plant_values)


@app.route('/harvest', methods=['POST'])
def predict_harvest():
    features_whitelist = ["FreshWeightShoot", "DryWeightShoot", "Height", "Diameter", "LeafArea"]
    form_values = [float(flask.request.form.get(field)) for field in features_whitelist]
    X = pd.DataFrame([form_values], columns=features_whitelist)
    variety = flask.request.form.get('variety')
    if variety is not None:
        X = data.process_plant_values(X, variety)
        print(X)
        prediction = harvest.predict(X)
    else:
        prediction = harvest_wo_variety.predict(X)

    # TODO date maken
    return flask.render_template('result.html', result_time=prediction)


if __name__ == '__main__':
    app.run()
