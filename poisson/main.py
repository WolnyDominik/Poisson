import numpy as np
import cv2

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, UpSampling2D, BatchNormalization, ReLU, ConvLSTM2D, Concatenate, Flatten, TimeDistributed, LSTM
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.models import Sequential, Model, save_model, load_model
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import relu, swish, softmax
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img, save_img
import tensorflow as tf

from os import listdir
from os.path import isfile, join

import tkinter as tk
from tkinter.constants import NW
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename


class Image_batcher():
    def __init__(self):
        self.bpp = 0
        self.ois = 0
        self.nis = 0
        self.size = 180
        self.offset = 10


    def batch_pic(self, path):
        img = img_to_array(load_img(path))
        x = img.shape[0]
        y = img.shape[1]
        self.ois = img.shape
        batches = []
        new_img = np.concatenate((img, img), axis=0)
        new_img = np.concatenate((new_img, new_img), axis=1)
        self.nis = new_img.shape
        self.bpp = (int(x/(self.size-2*self.offset))+1, int(y/(self.size-2*self.offset))+1)
        offset = self.offset * 2
        
        while new_img.shape[0] < 180:
            new_img = np.concatenate((new_img, new_img), axis=0)

        while new_img.shape[1] < 180:
            new_img = np.concatenate((new_img, new_img), axis=1)

        for i in range(self.bpp[0]):
            for j in range(self.bpp[1]):
                sx,ex,sy,ey = j*self.size , (j+1)*self.size , i*self.size , (i+1)*self.size

                if i == 0:
                    sx -= j*offset
                    ex -= j*offset
                elif j == 0:
                    sy -= i*offset
                    ey -= i*offset
                else:
                    sx -= j*offset
                    ex -= j*offset
                    sy -= i*offset
                    ey -= i*offset

                batches.append(new_img[sy:ey, sx:ex])

        return batches


    def unbatch_pic(self, batches, path):
        for i in range(self.bpp[0]):
            if i == 0:
                img_row = batches[i*self.bpp[1]][0 : self.size-self.offset , 
                                                 0 : self.size-self.offset]
            elif i == self.bpp[0]-1:
                img_row = batches[i*self.bpp[1]][self.offset : self.size , 
                                                 0 : self.size-self.offset]
            else:
                img_row = batches[i*self.bpp[1]][self.offset : self.size-self.offset , 
                                                 0 : self.size-self.offset]
        
            for j in range(1,self.bpp[1]):
                sx,ex,sy,ey = 0,self.size,0,self.size
                
                if i == 0 and j == self.bpp[1]-1:
                    sx = self.offset
                    ey -= self.offset
                elif i == 0:
                    sx = self.offset
                    ex -= self.offset
                    ey -= self.offset
                elif i == self.bpp[0]-1 and j == self.bpp[1]-1:
                    sx = self.offset
                    sy = self.offset
                elif i == self.bpp[0]-1:
                    sx = self.offset
                    ex -= self.offset
                    sy = self.offset
                elif j == self.bpp[1]-1:
                    sx = self.offset
                    sy = self.offset
                    ey -= self.offset
                else:
                    sx = self.offset
                    ex -= self.offset
                    sy = self.offset
                    ey -= self.offset

                img_row = np.concatenate((img_row, batches[i*self.bpp[1]+j][sy:ey,sx:ex]), axis=1)
            
            if i == 0:
                img = img_row
            else:
                img = np.concatenate((img, img_row), axis=0)
        
        save_img(path, array_to_img(img[0:self.ois[0],0:self.ois[1]]))


def loss_func(y_true, y_pred):
    return tf.nn.l2_loss(y_true - y_pred)


class Model():
    def __init__(self):
        self.load_model()
        self.batcher = Image_batcher()


    def apply_poisson(self, source_path: str, target_path: str, peak=70):
        image = cv2.imread(source_path, 1)
        noise_img = np.random.poisson(image / 255.0 * peak) / peak * 255
        cv2.imwrite(target_path, noise_img)


    def load_model(self):
        self.model = load_model(
            'model',
            custom_objects={"loss_func": loss_func}
        )


    def residual_block(self, x, i):
        res = x
        out = Conv2D(64, 3, padding='same', name=f'{i}_conv', activation=None)(x)
        out = BatchNormalization(name=f'{i}_batch')(out)
        out = ReLU(name=f'{i}_relu')(out)

        out = keras.layers.add([res, out])

        return out


    def create_model(self, image_size, d):
        inputs = Input(shape=image_size, name='input')
        x = Conv2D(64, (3,3), padding='same', name='in_conv', activation=None)(inputs)
        x = ReLU(name='in_relu')(x)

        for i in range(d):
            x = self.ResidualBlock(x, i)

        outputs = Conv2D(3, 3, padding='same', name='out_conv', activation=None)(x)

        model = Model(inputs, outputs, name='DnCNN')
        model.compile(
            optimizer='adam',
            loss=loss_func
        )

        return model


    def save_model(self):
        save_model(self.model, 'model', save_format='tf')

    
    def prepare_images(self):
        X_PATH = '/train_p/'
        Y_PATH = '/train/'

        files = [f for f in listdir(Y_PATH) if isfile(join(Y_PATH, f)) and f[f.rfind('.')+1:].lower() == 'jpg']

        y = []
        x = []

        for i, file in enumerate(files):
            if not isfile(join(X_PATH, file)):
                self.apply_poisson(join(Y_PATH, file), join(X_PATH, file))

            y += self.batcher.batch_pic(join(Y_PATH, file))
            x += self.batcher.batch_pic(join(X_PATH, file))
            print(i, '/', len(files))

        self.y_ds = np.array(y)
        self.x_ds = np.array(x)


    def train(self):
        self.prepare_images()
        self.model = self.create_model((180,180), 20)

        self.model.fit(
            x=self.x_ds,
            y=self.y_ds,
            epochs=50,
            batch_size=16,
            verbose=1
        )

        self.save_model()


    def predict(self, img_path):
        batches = self.batcher.batch_pic(img_path)
        
        preds = []
        for batch in batches:
            preds.append(self.model.predict(np.array([batch]), batch_size=1, verbose=1)[0])
        
        self.batcher.unbatch_pic(preds, 'clear.png')


class Window():
    def __init__(self):
        self.model = Model()
        self.window = tk.Tk()        
        self.window.geometry('630x350')
        self.window.resizable(0,0)
        self.window.title('Poisson denoiser')

        self.c_in = tk.Canvas(self.window, width=300,height=300)
        self.c_in.grid(column=0, row=0)

        self.c_out = tk.Canvas(self.window, width=300,height=300)
        self.c_out.grid(column=1, row=0)

        b1 = tk.Button(text='Choose input file...', width=15, height=1, command=self.open_file)
        b1.grid(column=0, row=1, pady=(10,10))

        b2 = tk.Button(text='Filter image', width=15, height=1, command=self.execute_nn)
        b2.grid(column=1, row=1, pady=(10,10))

        self.window.mainloop()

    def open_file(self):
        self.filepath = askopenfilename(
            filetypes=[("Images", "*.png")]
        )
        if not self.filepath:
            return
        else:
            self.load_img(self.filepath)

    def load_img(self, path: str):
        size = 300,300
        inimg = Image.open(path)
        inimg.thumbnail(size)
        img = ImageTk.PhotoImage(inimg)  
        self.c_in.create_image(20,20,anchor=NW, image=img)
        self.c_in.image = img

    def execute_nn(self):
        self.model.predict(self.filepath)
        inimg = Image.open('clear.png')
        size = 300,300
        inimg.thumbnail(size)
        img = ImageTk.PhotoImage(inimg) 
        self.c_out.create_image(20,20,anchor=NW, image=img)
        self.c_out.image = img


if __name__ == '__main__':
    w = Window()
