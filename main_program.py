
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from sklearn.preprocessing import LabelBinarizer


train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')



X_train = train.drop(['label'], axis = 1).values
X_test = test.drop(['label'], axis = 1).values

y_train = train['label']
y_test = test['label']

X_train = X_train.reshape(-1,28,28,1)
X_test = X_test.reshape(-1,28,28,1)


lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1, height_shift_range=0.1,
                                   shear_range=0.1,zoom_range=0.1)

datagen.fit(X_train)
generator = datagen.flow(X_train, y_train, batch_size= 128)

#Neural Network Creation
model = Sequential()
model.add(Conv2D(filters=128,kernel_size= 7, input_shape = (28,28,1), activation='relu' ))
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Conv2D(filters=128,kernel_size= 7))
model.add(MaxPool2D(pool_size=2,strides=2))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=24, activation='softmax'))

model.compile(optimizer = 'adam' , loss = 'categorical_crossentropy' , metrics = ['accuracy'])

history = model.fit(generator,
                   validation_data=(X_test,y_test),
                   epochs=1)



model.save('sign_detect.tf')
