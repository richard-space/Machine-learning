# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 20:25:38 2018

@author: 益慶
"""
import tkinter as tk
import numpy as np
import tensorflow as tf
import  matplotlib.pyplot as plt

data=np.zeros((94,3))
label=np.zeros((94,1))
inputdata=False
setNeu=False
### read data
def clickReadData():
    global E0text ,inputdata  
    try:
        with open(E0text.get()+'.txt','r') as f:
            j=0
            for line in f:
                fields=line.split("\t")
                data[j,]=int(fields[0]) ,int(fields[1]),int(fields[2])
                label[j]=int(fields[3])
                j=j+1
        print('Input data successfully')
        inputdata=True
    except:
        print('Please check your file path again')
    

x = tf.placeholder(tf.float32, [94, 3])
y = tf.placeholder(tf.float32, [94,1])

def setNeurons():
    global E1text, E2text, loss,optimizer,out_layer_addition,setNeu
    setNeu=True
    ### Network Parameters
    n_input = 3
    n_output = 1
    try:
        n_hidden_1 = int(E1text.get())        # 1st layer number of features
        n_hidden_2 = int(E2text.get())         # 2nd layer number of features
                 
        weights = {        
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))        
        }
        
        biases = {        
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        
            'out': tf.Variable(tf.random_normal([n_output]))        
        }
       
        ### Network structure
        
        # Hidden layer with RELU activation
        layer_1_multiplication = tf.matmul(x, weights['h1'])
        layer_1_addition = tf.add(layer_1_multiplication, biases['b1'])
        layer_1_activation = tf.nn.relu(layer_1_addition)
        
        # Hidden layer with RELU activation
        layer_2_multiplication = tf.matmul(layer_1_activation, weights['h2'])
        layer_2_addition = tf.add(layer_2_multiplication, biases['b2'])
        layer_2_activation = tf.nn.relu(layer_2_addition)
        
        # Output layer with linear activation
        out_layer_multiplication = tf.matmul(layer_2_activation, weights['out'])
        out_layer_addition = out_layer_multiplication + biases['out']
            
        loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=y,predictions=out_layer_addition))
        
        optimizer = tf.train.AdamOptimizer(learning_rate=0.05).minimize(loss)
        print('Upadte Neurons')
    except:
        print('Neurons number must be natural numbers') 

training_epochs = 1000

def clickGenerate():
    global inputdata,setNeu
    if inputdata==True and setNeu:       
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer()) #inits the variables 
            for epoch in range(training_epochs):
                c,_ = sess.run([loss,optimizer], feed_dict={x: data, y:label})
                if epoch % 100 == 0:
                    print('loss : {:.4f}'.format(c))
               
            # plot    
            z=sess.run(out_layer_addition, feed_dict={x: data, y:label})
            plt.plot(label,"ro")
            plt.plot(z)
            plt.plot(z,"ko")
            plt.show()
    else:
         print('Please input file and set Neurons')



#### GUI parts
root=tk.Tk()
root.title('Deep Learning')
F1=tk.Frame(root)
F1.pack()
F2=tk.Frame(root)
F2.pack()
F3=tk.Frame(root)
F3.pack()
L0=tk.Label(F1,text='Input Data File')
L0.pack(side='left')
E0text=tk.StringVar()
E0text.set('inputData')
E0=tk.Entry(F1,textvariable=E0text)
E0.pack(side='left')
B0=tk.Button(F1,text='Read Data',command=clickReadData)
B0.pack(side='left')
L1=tk.Label(F2,text='Layer1 =')
L1.pack(side='left')
E1text=tk.StringVar()
E1text.set('3')
E1=tk.Entry(F2,textvariable=E1text)
E1.pack(side='left')
L2=tk.Label(F2,text='Layer2 =')
L2.pack(side='left')
E2text=tk.StringVar()
E2text.set('2')
E2=tk.Entry(F2,textvariable=E2text)
E2.pack(side='left')
B1=tk.Button(F2,text='Update Neurons',command=setNeurons)
B1.pack(side='left')
B2=tk.Button(F3,text='Generate',command=clickGenerate)
B2.pack(side='left')
root.mainloop()