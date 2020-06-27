from keras.models import Sequential, Model
from keras.models import load_model
from keras.layers import BatchNormalization, MaxPooling2D, Dropout, Conv2D, Input
from keras.layers import Flatten, Dense
from keras import optimizers
# from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam


def create_model():
    '''
    网络的输入为96x96的单通道灰阶图像, 输出30个值, 代表的15个关键点的横坐标和纵坐标
    '''
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(96, 96, 1),
                     activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (2, 2),  activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (2, 2),  activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(30))


    # # 使用ZFnet
    # img_height, img_width, channels = 96, 96, 1
    # inputs = Input((img_height, img_width, channels))
    # # 第一个卷积层，96个卷积核，大小7x7，步长2，不加边，激活relu
    # c1 = Conv2D(96, (7, 7), strides=(2, 2),  # kernel size 11-->7; strides 4-->2
    #             padding='same', activation='relu',
    #             kernel_initializer='uniform')(inputs)
    # # 原文中是LRN，这里用BN代替
    # # c2 = BatchNormalization()(c1)
    # # 池化，核大小3x3，步长2，不加边
    # c3 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(c1)
    #
    # # c4-c6类似c1-c3，核数量、大小、步长、加边有变化
    # c4 = Conv2D(256, (5, 5), strides=(2, 2), padding='same',  # strides 1-->2
    #             activation='relu', kernel_initializer='uniform')(c3)
    # # c5 = BatchNormalization()(c4)
    # c6 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(c4)
    #
    # # c7、c8两个卷积层一样
    # c7 = Conv2D(384, (3, 3), strides=1, padding='same',
    #             activation='relu', kernel_initializer='uniform')(c6)
    # c8 = Conv2D(384, (3, 3), strides=1, padding='same',
    #             activation='relu', kernel_initializer='uniform')(c7)
    # # 再接一个c9卷积层
    # c9 = Conv2D(256, (3, 3), strides=1, padding='same',
    #             activation='relu', kernel_initializer='uniform')(c8)
    # # 池化
    # c10 = MaxPooling2D((3, 3), strides=2, padding='same')(c9)
    #
    # # 展平后接全2个连接成，分别连有一个dropout层
    # c11 = Flatten()(c10)
    # c12 = Dense(4096, activation='relu')(c11)
    # c13 = Dropout(0.5)(c12)
    #
    # c14 = Dense(4096, activation='relu')(c13)
    # c15 = Dropout(0.5)(c14)
    #
    # # 输出30个值, 代表的15个关键点的横坐标和纵坐标
    # outputs = Dense(30)(c15)
    #
    # # 编译与训练
    # model = Model(inputs=[inputs], outputs=[outputs])

    return model


def compile_model(model):
    # optimizer = 'SGD'
    loss = 'mean_squared_error'
    metrics = ['mean_squared_error']
    model.compile(optimizer=optimizers.SGD(lr=0.01, momentum=0.9), loss=loss, metrics=metrics)


def train_model(model, X_train, y_train):
    return model.fit(X_train, y_train, epochs=25, batch_size=64, verbose=2, validation_split=0.2)


def save_model(model, fileName):
    model.save(fileName + '.h5')


def load_trained_model(fileName):
    return load_model(fileName + '.h5')

