import random

import flask

from Flask_API.tabular_predictor import tabular_predictor
from data_processing import Data
import tensorflow as tf
import numpy as np

app = flask.Flask(__name__, template_folder='./templates')

print('Loading models...')

# TODO: Load these from disk
# diameter = tf.keras.models.load_model('../Plant_variable_predictor/checkpoints/rgb_diameter')
diameter = tf.keras.models.load_model(r"../Plant_variable_predictor/checkpoints/rgb_augm_reg_diameter/cp-0029.ckpt")

# height = tf.keras.models.load_model('../Plant_variable_predictor/checkpoints/depth_height')
# height = tf.keras.models.load_model(r"../Plant_variable_predictor/checkpoints/depth_height/cp-0088.ckpt")

leaf_area = tf.keras.models.load_model("../Plant_variable_predictor/checkpoints/rgb_augm_reg_leafarea/cp-0035.ckpt")
# fresh_weight = tf.keras.models.load_model('../Plant_variable_predictor/checkpoints/rgb-d_freshweight')

# dry_weight = tf.keras.models.load_model('../Plant_variable_predictor/checkpoints/rgb_dryweight')


print('Models loaded.')


@app.route('/', methods=['GET', 'POST'])
def index():
    return flask.render_template('home.html')

@app.route('/image', methods=['POST'])
def predict_image():

    data = Data()

    rgb_img_req = flask.request.files.get("rgb")
    depth_img_req = flask.request.files.get("depth")

    # Both files should be uploaded
    # TODO: Check if this is how it is intended
    assert rgb_img_req is not None
    assert depth_img_req is not None

    rgb_bytes_img = rgb_img_req.read()
    depth_bytes_img = depth_img_req.read()

    prepared_rgb_img = data.prepare_image(image=rgb_bytes_img,
                                          image_type="rgb",
                                          requires_normalization=0
                                          )
    # prepared_depth_img = data.prepare_image(image=rgb_bytes_img,
    #                                         image_depth=depth_bytes_img,
    #                                         image_type="rgbd",
    #                                         requires_normalization=0
    #                                         )



    # prepared_rgb_img_normalized = data.prepare_image(image=rgb_bytes_img,
    #                                       image_type="rgb",
    #                                       requires_normalization=True
    #                                       )
    # prepared_depth_img_normalized = data.prepare_image(image=depth_bytes_img,
    #                                         image_type="rgbd",
    #                                       requires_normalization=True
    #                                         )

    




    print(prepared_rgb_img.shape)
    pred1 = leaf_area.predict(prepared_rgb_img)
    print(pred1)


    # plant_values = {'diameter': diameter.predict(rgb),
    #                 'height': height.predict(depth),
    #                 'leaf_area': leaf_area.predict(rgb),
    #                 'freshweight': fresh_weight.predict(np.dstack((rgb.numpy(), depth.numpy()))),
    #                 'dryweight': dry_weight.predict(rgb)}
    
    # TODO Remove test data and replace it with functioning values
    plant_values = {'diameter': random.randint(1,9999),
                    'height': random.randint(1,9999),
                    'leaf_area': random.randint(1,9999),
                    'freshweight': random.randint(1,9999),
                    'dryweight': random.randint(1,9999)}
    return flask.render_template('extraction.html', result_plant=plant_values)


@app.route('/harvest', methods=['POST'])
def predict_harvest():
    features_whitelist = [ "Height", "Diameter", "LeafArea", "FreshWeightShoot", "DryWeightShoot"] # + ["Variety"] # Variety is not (yet) included in frontend
    form_values = [flask.request.form.get(field) for field in features_whitelist]
    prediction = tabular_predictor.predict(form_values)

    # TODO: This does not seem to be the right format, if so, format it correctly
    predicted_date = prediction[0]

    # predicted_date = '11/09/2001'
    return flask.render_template('result.html', result_time=predicted_date)

if __name__ == '__main__':
    app.run()
