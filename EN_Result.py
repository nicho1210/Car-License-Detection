#using car_license_model.h5 to  detect the number of 'word_5' to 'word_8' in the image

# 1、导入模块
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# 2、加载模型
model_en = load_model('car_license_model_en.h5')


# Read the image
#img = cv2.imread(f'dataset-cover.jpg')
img = cv2.imread(f'word_8_whiten.jpg')
img = cv2.resize(img, (28, 28))
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = img / 255.0
img = np.expand_dims(img, axis=0)
img = np.expand_dims(img, axis=3)
cv2.imwrite(f'dataset-cover_res.jpg', img)

# Predict
result = model_en.predict_step(img)

# Output the prediction result
print(f"Prediction for dataset-cover.jpg: {np.argmax(result)}")
