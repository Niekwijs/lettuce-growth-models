# wacky af
import os; os.system("pip install -r requirements.txt")

import pickle
import flask
import pandas as pd

from data_processing import Data
import tensorflow as tf
import numpy as np
import datetime


ENVIRONMENT = "dev"

app = flask.Flask(__name__, template_folder='./templates')


# base_path = "./keras_models"
base_path = "/modelstoragedrive/"

loaded = False

models = {
    "fresh_weight":{
        "name": "fresh_weight",
        "path": "rgb-d_freshweight/cp-0100.ckpt",
        "loading":False,
        "model":None,
        "type":"keras"
    },
    "diameter":{
        "name": "diameter",
        "path": "rgb_augm_reg_diameter/cp-0029.ckpt",
        "loading":False,
        "model":None,
        "type":"keras"
    },
    "height":{
        "name": "height",
        "path":"depth_reg_height/cp-0098.ckpt",
        "loading": False,
        "model": None,
        "type":"keras"
    },
    "leaf_area":{
        "name": "leaf_area",
        "path":"rgb_augm_reg_leafarea/cp-0035.ckpt",
        "loading": False,
        "model": None,
        "type":"keras"
    },
    "dry_weight":{
        "name": "dry_weight",
        "path":"rgb_dryweight/cp-0077.ckpt",
        "loading": False,
        "model": None,
        "type":"keras"
    },
    "harvest":{
        "name": "harvest",
        "path": "Forecasting/linear_model-all-varieties.sav",
        "loading": False,
        "model": None,
        "type": "pickle"
    },
    "harvest_wo_variety":{
        "name": "harvest_wo_variety",
        "path": "Forecasting/linear_model-all-varieties-no-dum.sav",
        "loading": False,
        "model": None,
        "type": "pickle"
    }
}



# @app.before_first_request
def load_models():
    global models, loaded
    print("Started loading models...")
    for _,model in models.items():
        if model["model"] is None and not model["loading"]:
            print(f"Started loading model: {model['name']} -> {model['model']} ")
            model['loading'] = True
            if model["type"] == "keras":
                model["model"] = tf.keras.models.load_model(base_path + model["path"]) # This should ideally be done async, but probably not enough memory
            elif model["type"] == "pickle":
                model["model"] = pickle.load(open(base_path + model["path"], "rb"))
            print(f"Finished loading model: {model['name']} -> {model['model']} ")

    loaded = True
    print('All models loaded.')




data = Data()


@app.route('/', methods=['GET', 'POST'])
def index():
    # app.before_first_request does not seem to work propperly, so cold start on first request
    global loaded
    if not loaded:
        load_models()

    return flask.render_template('home.html',
                                 fresh_weight_loaded=models["fresh_weight"]["loading"],
                                 diameter_loaded=models["diameter"]["loading"],
                                 height_loaded=models["height"]["loading"],
                                 leaf_area_loaded=models["leaf_area"]["loading"],
                                 dry_weight_loaded=models["dry_weight"]["loading"],
                                 harvest_loaded=models["harvest"]["loading"],
                                 harvest_wo_variety_loaded=models["harvest_wo_variety"]["loading"])


@app.route('/image', methods=['POST'])
def predict_image():
    global models
    rgb_img_req = flask.request.files.get("rgb")
    depth_img_req = flask.request.files.get("depth")

    rgb_bytes_img = rgb_img_req.read()
    depth_bytes_img = depth_img_req.read()

    normalized_images = data.prepare_images(image=rgb_bytes_img,
                                            image_depth=depth_bytes_img,
                                            requires_normalization=True)

    not_normalized_images = data.prepare_images(image=rgb_bytes_img,
                                                image_depth=depth_bytes_img,
                                                requires_normalization=False)


    plant_values = {'diameter': models["diameter"]["model"].predict(normalized_images['rgb'])[0][0],
                    'height': models["height"]["model"].predict(normalized_images['depth'])[0][0],
                    'leaf_area':  models["leaf_area"]["model"].predict(normalized_images['rgb'])[0][0],
                    'dryweight':  models["dry_weight"]["model"].predict(not_normalized_images['rgb'])[0][0],
                    'freshweight': models["fresh_weight"]["model"].predict(
                        np.concatenate((not_normalized_images['rgb'], not_normalized_images['depth']), axis=-1))[0][0],
                    }

    print(plant_values['diameter'])
    return flask.render_template('extraction.html', result_plant=plant_values)


@app.route('/harvest', methods=['POST'])
def predict_harvest():
    features_whitelist = ["FreshWeightShoot", "DryWeightShoot", "Height", "Diameter", "LeafArea"]
    form_values = [float(flask.request.form.get(field)) for field in features_whitelist]
    X = pd.DataFrame([form_values], columns=features_whitelist)
    variety = flask.request.form.get('variety')
    if variety in data.varieties:
        X = data.process_plant_values(X, variety)
        prediction = models["harvest"]["model"].predict(X)
    else:
        prediction = models["harvest"]["model"].predict(X)

    now = datetime.datetime.now()
    predicted_date = now + datetime.timedelta(weeks=(data.max_weeks - prediction[0]))
    predicted_date_formatted = predicted_date.strftime("%d/%m/%Y")
    return flask.render_template('result.html', result_time=predicted_date_formatted)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000 if ENVIRONMENT == "dev" else 80)
