#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: KunalDas
"""

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class dogcat:
    def __init__(self,filename):
        self.filename =filename


    def predictiondogcat(self):
        # load model
        model = load_model('model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'Diya'
            return [{"image": prediction}]
        elif result[0][1] == 1:
            prediction = 'Kunal'
            return [{"image": prediction}]
        else:
            prediction = 'Smriti'
            print(prediction)
            return [{"image": prediction}]

