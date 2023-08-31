import data_process
import tensorflow as tf
import keras
from keras.layers.advanced_activations import PReLU
from sklearn import metrics
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import LambdaCallback
import numpy as np
import pandas as pd
from pathlib import Path

def calculate_acc(agg_scores,agg_y_true):
    #agg_scores = np.asarray(agg_scores)
    #agg_y_true = np.asarray(agg_y_true)
    #avg_loss = cross_entropy_error(agg_scores,agg_y_true)
    #print(avg_loss)
    agg_scores[agg_scores<0.5]=0
    agg_scores[agg_scores>=0.5]=1
    agg_roc_auc = metrics.confusion_matrix(agg_y_true, agg_scores)
    c0 = agg_roc_auc[0]
    c1 = agg_roc_auc[1]
    avg_acc = (c0[0]/(c0[0]+c0[1])+c1[1]/(c1[0]+c1[1]))/2
    return avg_acc, agg_roc_auc

def print_wrong_name(agg_scores,agg_y_true):
    agg_scores = np.asarray(agg_scores)
    agg_y_true = np.asarray(agg_y_true)
    #avg_loss = cross_entropy_error(agg_scores,agg_y_true)
    #print(avg_loss)
    agg_scores[agg_scores<0.5]=0
    agg_scores[agg_scores>=0.5]=1
    agg_roc_auc = metrics.confusion_matrix(agg_y_true, agg_scores)
    c0 = agg_roc_auc[0]
    c1 = agg_roc_auc[1]
    avg_acc = (c0[0]/(c0[0]+c0[1])+c1[1]/(c1[0]+c1[1]))/2
    return avg_acc, agg_roc_auc

class MyCallback(keras.callbacks.Callback):
    def __init__(self,x_train, y_train,labels_train, x_test, y_test, labels_test, x_val, y_val, labels_val, save_dir):
        self.test_accs=[]  
        self.train_accs=[] 
        self.val_accs=[] 
        self.epoches=[]
        self.epoch=0
        self.x_train=x_train
        self.y_train=y_train
        self.labels_train=labels_train
        self.x_test=x_test
        self.y_test=y_test
        self.labels_test=labels_test
        self.x_val=x_val
        self.y_val=y_val
        self.labels_val=labels_val
        self.save_dir=save_dir
    def on_epoch_end(self, batch, logs={}): 
        self.epoch_df = pd.DataFrame()
        self.epoch+=1
        self.epoches.append(self.epoch)
        p_test = self.model.predict(self.x_test)
        p_val = self.model.predict(self.x_val)
        p_train = self.model.predict(self.x_train)
        #print(np.squeeze(p_test,1).shape)
        #print(y_test.shape)
        test_acc, test_cm = calculate_acc(np.squeeze(p_test,1), self.y_test)
        train_acc, train_cm = calculate_acc(np.squeeze(p_train,1), self.y_train)
        val_acc, val_cm = calculate_acc(np.squeeze(p_val,1), self.y_val)
        self.test_accs.append(test_acc)
        self.train_accs.append(train_acc)
        self.val_accs.append(val_acc)
        print(f"[train] acc_score: {train_acc:.4f}, confusion_matrix:")
        print(f"{train_cm} ")
        print(f"[val] acc_score: {val_acc:.4f}, confusion_matrix: ")
        print(f"{val_cm}, ")
        print(f"[test] acc_score: {test_acc:.4f}, confusion_matrix: ")
        print(f"{test_cm}")
        self.epoch_df = self.epoch_df.append(pd.DataFrame({'epoch': self.epoches,
                                                 'train_acc': self.train_accs,
                                                 'val_acc': self.val_accs,
                                                 'test_acc': self.test_accs}))
        self.epoch_df = self.epoch_df[['epoch', 'train_acc', 'val_acc', 'test_acc']]

        self.epoch_df.to_csv(Path(self.save_dir, f'result_{an_id:02}.csv'), index=False)
        return






def train(lr, an_id, clip_size, stride):




    x_train, y_train,labels_train, x_test, y_test, labels_test, x_val, y_val, labels_val = data_process.load_data(an_id, stride, clip_size)






    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(7,clip_size,18)))
    model.add(tf.keras.layers.Dense(50, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dense(100))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    adamOptimizer = tf.optimizers.Adam(learning_rate = lr)

    fold_dir = '/home/fanxuan/SMNN/result/'
    save_dir = Path(fold_dir, f'without_43_1_7_lr_{lr}_clip{clip_size}_stride_{stride}_all')
    save_dir.mkdir(parents=True, exist_ok=True)

    epoch_print_callback = MyCallback(x_train, y_train,labels_train, x_test, y_test, labels_test, x_val, y_val, labels_val, save_dir)
    model.compile(adamOptimizer,
              loss='binary_crossentropy',
              metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=64, epochs=100, validation_data=(x_test, y_test), shuffle=True, class_weight ={0:1,1:7} , callbacks=[epoch_print_callback])
    model.summary()
    return

lr = 0.00005
clip_size = 200
stride = 40
for i in range(5):
    an_id = i
    train(lr, an_id, clip_size, stride)
'''lr = 0.000001
clip_size = 200
stride = 40
for i in range(5):
    an_id = i
    train(lr, an_id, clip_size, stride)'''
#print(model.predict(x_test))
#print(model.predict(x_test).shape)
#print(y_test.shape)
'''#fashion_mnist = keras.datasets.fashion_mnist
#devide the data into training and test:train 60000 and test 10000
(train_images, train_lables),(test_images, test_lables) = fashion_mnist.load_data()
 
 

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), 
    keras.layers.Dense(128, activation= tf.nn.relu),    
    keras.layers.Dense(10, activation = tf.nn.softmax)  
])
 
# train and evaluate the module

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(train_images_scaled, train_lables, epochs=5)
 
test_images_scaled = test_images/255
model.evaluate(test_images_scaled, test_lables)
 
#module to predict
print(model.predict(test_images/255)[0])'''

'''
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(train_images_scaled, train_lables, epochs=5)
 
test_images_scaled = test_images/255
model.evaluate(test_images_scaled, test_lables)
 
#module to predict
print(model.predict(test_images/255)[0])'''
