from django.apps import AppConfig
import os
from tensorflow.keras.models import model_from_json
import json
from django.conf import settings


IMAGE_SIZE = [256, 256]


class BmiConfig(AppConfig):
    name = 'bmi'
    WEIGHTS = os.path.join(
        settings.MODEL_DIR, "vgg_faces_MAE_VAL-3.082510232925415.h5")

    MODEL = os.path.join(settings.MODEL_DIR, 'model_data.json')

    with open(MODEL, 'r') as f:
        json_data = json.load(f)

    predictor = model_from_json(json_data)
    predictor.load_weights(WEIGHTS)
    # predictor.load(PRETRAINED_MODEL_PATH)
