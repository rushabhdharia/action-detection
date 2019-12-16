from tensorflow.keras import Model
from tensorflow.keras.layers import Conv3D, BatchNormalization, MaxPool3D, ConvLSTM2D, MaxPool2D, LayerNormalization, Dense, Flatten 


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv3d_1 = Conv3D(2, kernel_size = (1,7,7), padding = 'same')
        self.batchnorm_1 = BatchNormalization()
        self.batchnorm_2 = BatchNormalization()
        self.batchnorm_3 = BatchNormalization()
        self.maxpool3d = MaxPool3D(pool_size=(1,2,2))
        self.conv3d_2 = Conv3D(4, kernel_size = (1,5,5), padding = 'same')
        self.conv3d_3 = Conv3D(8, kernel_size = (1,3,3), padding = 'same')


    def call(self, inputs):
        x = self.conv3d_1(inputs)
        x = self.batchnorm_1(x)
        x = self.maxpool3d(x)
        x = self.conv3d_2(x)
        x = self.batchnorm_2(x)
        x = self.maxpool3d(x)
        x = self.conv3d_3(x)
        x = self.batchnorm_3(x)
        x = self.maxpool3d(x)
        return x

class MyCL_Model(Model):
    
    def __init__(self):
        super(MyCL_Model, self).__init__()
        self.encoder = Encoder()
        self.convlstm_1 = ConvLSTM2D(filters=16, kernel_size=(3, 3), strides = (2, 2) ,padding='valid', return_sequences=False)
        self.layernorm = LayerNormalization()
        self.maxpool2d = MaxPool2D()
        self.flatten = Flatten()
        self.dense_1 = Dense(100, activation = 'relu')
        self.classifier = Dense(6, activation = 'softmax') 

        
    def call(self, inputs):
        x = self.encoder(inputs)
        x = self.convlstm_1(x)
        x = self.layernorm(x)
        x = self.maxpool2d(x)
        x = self.flatten(x)
        x = self.dense_1(x)

        return self.classifier(x)