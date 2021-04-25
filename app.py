from flask import Flask, request, render_template
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import tensorflow as tf

app = Flask(__name__)


savedmodel = load_model('sign_detect.tf')

#savedmodel.summary()
@app.route('/', methods=['GET', 'POST'])
def predict_sign():
	if request.method == 'GET':
		return render_template('index.html', value='hi')
	if request.method == 'POST':
		print(request.files)
		if 'file' not in request.files:
			print('file not uploaded')
			return
		picfile = request.files['file']
		test_image = image.load_img(picfile.filename, target_size = (28,28))
		test_image = image.img_to_array(test_image)
		test_image = tf.image.rgb_to_grayscale(test_image)
		test_image = np.expand_dims(test_image, axis=0)
		result = savedmodel.predict(test_image)
		sign_index =  np.argmax(result)
		return render_template('result.html', sign=sign_index)

if __name__ == '__main__':
	app.run(debug=True)