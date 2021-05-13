import logging
import os
import re

import tensorflow as tf
import tensorflow.contrib.slim as slim
from lib.model import data_loader, generator, SRGAN, test_data_loader, inference_data_loader, save_images, SRResnet
from lib.ops import *
import math
import time
import numpy as np

from PIL import Image
from sagemaker_inference import content_types
from sagemaker_inference import encoder
from six import BytesIO


logging.basicConfig(format="%(asctime)s %(message)s", level=logging.INFO)

class SrganTf:
    def __init__(self, model):
        self.model = model

    def __init__(self, model):
        self.model = model
def model_fn(model_dir):
    """Create our inference task as a delegate to the model. This runs only once per one worker"""
    for root, dirs, files in os.walk(model_dir):
        for file in files:
            if re.compile(".*\\.pt").match(file):
                checkpoint = re.compile(".*\\.pt").match(file).group()
    try:
        model = torch.load(checkpoint)
        if torch.cuda.is_available():
            model.to("cuda")
        model.eval()
        return PyTorchIC(model=model)
    except Exception:
        logging.exception("Failed to load model")
        raise


def transform_fn(task: SrganTF, input_data, content_type, accept):
    """Make predictions against the model and return a serialized response.
    The function signature conforms to the SM contract

    Args:
        task (obj): model loaded by model_fn, in our case is one of the Task.
        input_data (obj): the request data.
        content_type (str): the request content type.
        accept (str): accept header expected by the client.

    Returns:
        obj: the serialized prediction result or a tuple of the form
            (response_data, content_type)

    """
    # input_data = decoder.decode(input_data, content_type)
    if content_type == "application/x-image":
        input_data = Image.open(BytesIO(input_data)).convert("RGB")
        try:
            output = task(input_data=input_data, content_type=content_type, accept=accept)
            return output
        except Exception:
            logging.exception("Failed to do transform")
            raise
    raise ValueError('{{"error": "unsupported content type {}"}}'.format(content_type or "unknown"))
