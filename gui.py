from tkinter import *
from tkinter import filedialog
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img
import keras
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


model = load_model('model.h5')
opt = keras.optimizers.SGD(lr = 0.001)
model.compile(loss='categorical_crossentropy',
              optimizer = opt,
              metrics=['accuracy'])
root = Tk()
root.geometry('500x400+300+200')


testDataGen = ImageDataGenerator(rotation_range = 45.0,
                                                      zoom_range = 0.3)
testGenerator = testDataGen.flow_from_directory( 'data/test', target_size = (20, 20),
                                                                        color_mode = 'rgb')#,seed = 777,
                                                                        #save_to_dir = 'generated') 

def fileChoose():
    root.filename = filedialog.askopenfilename(initialdir = "", title = "Select image")
    return
    
def plotDiagram(predictions):
    #print(prediction)
    #print(prediction.shape)
    fig, ax = plt.subplots()
    ind = np.arange(1, 18)
    prediction = predictions.ravel()
    
    af = plt.bar(1, prediction[0])
    al = plt.bar(2, prediction[1])
    alg = plt.bar(3, prediction[2])
    an = plt.bar(4, prediction[3])
    ang = plt.bar(5, prediction[4])
    ant = plt.bar(6, prediction[5])
    ar = plt.bar(7, prediction[6])
    arm = plt.bar(8, prediction[7])
    au = plt.bar(9, prediction[8])
    aus = plt.bar(10, prediction[9])
    az = plt.bar(11, prediction[10])
    bh = plt.bar(12, prediction[11])
    bhr = plt.bar(13, prediction[12])
    ba = plt.bar(14, prediction[13])
    bar = plt.bar(15, prediction[14])
    be = plt.bar(16, prediction[15])
    bel = plt.bar(17, prediction[16])
    ax.set_xticks(ind)
    ax.set_xticklabels(['Afghanistan', 'Albania', 'Algeria', 'Andorra', 'Angola', 'Antigua and Barbuda',
                                'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain',
                                'Bangladesh', 'Barbados', 'Belarus', 'Belize'])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show(block = False)

predicted = np.array([])

def randomSample():
                                                         
    x, y = testGenerator.next()
    #print(x[0].shape)
    print(np.argmax(y[0]))
    y_pred = model.predict(x)
    print(np.argmax(y_pred[0]))
    print('-----------')
    #print(x[0])
    plt.imshow(x[0].astype('uint8'))
    plt.show()
    
def predict():
    if root.filename :
        img = Image.open(root.filename)
        img = img.resize((20, 20))
        #img.save("blabla,jpg", "JPEG")
        arr = np.array((img_to_array(img)),)
        arr = arr.reshape((1, 20, 20, 3))
        #print(type(arr[0][0][0][0]))
        #arr /= 255.
        predicted = model.predict(arr)
        predicted = np.around(predicted,decimals = 2)
        plotDiagram(predicted)
        #print(root.filename)
        #bla = 'conv2d_2'
        #middle_model = Model(inputs = model.input, outputs=model.get_layer(bla).output)
        #output = middle_model.predict(arr)
        #print(np.around(output, decimals = 2))
        print(predicted)

#test()
button = Button(root, text=u"Choose File", command=fileChoose, width = 15, height = 10)
button1 = Button(root, text=u"Predict", command=predict, width = 15, height = 10)
button2 = Button(root, text=u"Random Sample", command=randomSample, width = 15, height = 10)

button.pack(side = "top", expand = True)
button2.pack(expand = True)
button1.pack(side = "bottom", expand = True)

root.mainloop()