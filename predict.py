import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from flask import render_template
import io

class blooddetection:

    def bloodgroup(self, image_file):

        self.image_file = image_file

        # load the model
        model = load_model('CNN.h5')

        #Save the file to ./uploads

        basepath = os.path.dirname(__file__)

        file_path = os.path.join(basepath, 'uploads', secure_filename(self.image_file.filename)) 

        
        self.image_file.save(file_path)  # save the image for further use - org

        test_image = image.load_img(file_path, target_size=(224, 224)) # should be same as given in the code for input
        test_image = image.img_to_array(test_image)
        test_image = test_image / 255
        test_image = np.expand_dims(test_image, axis=0)  # expand dimension - flattening it

        preds = model.predict(test_image)
    

        preds = np.argmax(preds,axis=1)  # The numpy. argmax() function returns indices of the max element of the array in a particular axis.
   

        if preds == 0:
            prediction = "A+"
            return prediction
        elif preds == 1:
            prediction = "A-"
            return prediction
        elif preds == 2:
            prediction = "AB+"
            return prediction
        elif preds == 3:
            prediction = "AB-"
            return prediction
        elif preds == 4:
            prediction = "B+"
            return prediction
        elif preds == 5:
            prediction = "B-"
            return prediction
        elif preds == 6:
            prediction = "O+"
            return prediction
        elif preds == 7:
            prediction = "O-"
            return prediction
        
      