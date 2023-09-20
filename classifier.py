from save_load import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import(Conv2D,MaxPooling2D,Dense,Dropout,Flatten,LSTM)
import numpy as np
from confusion_matrix import *
import numpy as np
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train=load('X_train')
X_test=load('X_test')
Y_train=load('Y_train')
Y_test=load('Y_test')

def cnn(X_train,X_test,Y_train,Y_test):

    X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1,1))
    X_test=X_test.reshape((X_test.shape[0],X_test.shape[1],1,1))

    model=Sequential()
    model.add(Conv2D(64,(1,1),padding='valid',input_shape=X_train[1].shape,activation='relu'))
    model.add(MaxPooling2D(pool_size=(1,1)))
    model.add(Dense(2,activation='softmax'))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=2, batch_size=400, verbose=0)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    return y_predict

def ann(X_train,X_test,Y_train,Y_test):

    model = Sequential()
    model.add(Dense(20, activation='softmax'))
    model.add(Dense(10, activation='softmax'))
    model.add(Dense(2, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=2, batch_size=100, verbose=0)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    return y_predict



def pro_classifier():
    pred1 = cnn(X_train,X_test,Y_train,Y_test)
    save('cnn_y_pred',pred1)

    pred2 = ann(X_train,X_test,Y_train,Y_test)


    #pred1=np.array(pred1)
    #save('pred1',pred1)
    #pred2=np.array(pred1)
    #save('pred2',pred2)

pro_classifier()