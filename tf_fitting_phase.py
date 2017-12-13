# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:20:34 2017

@author: pyh
"""

def read_data(file_name):
    import xlrd
    read_x=[]
    read_y=[]
    data=xlrd.open_workbook(file_name)
    table=data.sheets()[0]
    num=0
    for i in range(1,table.nrows):
        for j in range(1,table.ncols):
            if table.row(i)[j].value != '':
                read_x.append([])
                read_x[num].append(table.row(i)[0].value)
                read_x[num].append(table.row(0)[j].value)
                read_y.append([0,0,0,0,0])
                read_y[num][int(table.row(i)[j].value)]=1
                num+=1        
    return read_x,read_y
    

#def get_test(x,y,n=50):
#    x_test=[]
#    y_test=[]
#    selec=[]
#    while len(selec) < n:
#        i=int(np.random.uniform(low=0,high=len(x)-1))
#        if i not in selec:
#            selec.append(i)
#            x_test.append(x[i])
#            y_test.append(y[i])
#    return x_test,y_test

def get_batch(xs,ys,n=20):
    x=xs.copy()
    y=ys.copy()
    m=int(len(x)/n)
    x_batch=[]
    y_batch=[]
    for i in range(m):
        x_batch.append([])
        y_batch.append([])
        for j in range(n):
            selec=int(np.random.uniform(low=0,high=len(x)-1))
            x_batch[i].append(x[selec])
            y_batch[i].append(y[selec])
            del x[selec]
            del y[selec]
    if x !=[]:
        x_batch.append(x)
        y_batch.append(y)
#强调0,7的0相
    for j in range(8):
        x_batch.append([])
        y_batch.append([])
        for i in range(20):
            x_batch[len(x_batch)-1].append([-0.03,7])
            y_batch[len(y_batch)-1].append([0,1,0,0,0])
        for i in range(20):
            x_batch[len(x_batch)-1].append([0.0,7.1])
            y_batch[len(y_batch)-1].append([1,0,0,0,0])
        for i in range(20):
            x_batch[len(x_batch)-1].append([0.03,7.0])
            y_batch[len(y_batch)-1].append([0,1,0,0,0])
        for i in range(20):
            x_batch[len(x_batch)-1].append([0.0,7.2])
            y_batch[len(y_batch)-1].append([1,0,0,0,0])
    return x_batch[1:len(x_batch)],y_batch[1:len(y_batch)],x_batch[0],y_batch[0]


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
x=tf.placeholder(tf.float32,[None,2])
y_=tf.placeholder(tf.float32,[None,5])
prob=tf.placeholder(tf.float32)

hidden_num=8

Weights_1=tf.Variable(tf.random_normal([2,hidden_num],dtype=tf.float32))
bias_1=tf.Variable(tf.zeros([hidden_num],dtype=tf.float32))
hidden_1=tf.nn.dropout(tf.nn.softmax(tf.matmul(x,Weights_1)+bias_1),keep_prob=prob)

Weights_2=tf.Variable(tf.random_normal([hidden_num,5],dtype=tf.float32))
bias_2=tf.Variable(tf.zeros([5],dtype=tf.float32))
y=tf.nn.softmax(tf.matmul(hidden_1,Weights_2)+bias_2)

loss=tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y),reduction_indices=[1]))
optimizer=tf.train.AdamOptimizer(0.01).minimize(loss)

acurracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),dtype=tf.float32))
phase_get=tf.reduce_mean(tf.argmax(y,1))

init=tf.global_variables_initializer()
sess=tf.InteractiveSession()

sess.run(init)

xs,ys=read_data("3L.xlsx")
accu1=0
accu2=0
node=1
sumnum=0
while accu1<0.92 or accu2<0.95:
    x_batch,y_batch,x_test,y_test=get_batch(xs,ys)
#    x_test,y_test=get_test(xs,ys)
#    sess.run(optimizer,feed_dict={x:xs,y_:ys,prob:1})   
    for j in range(len(x_batch)):              
        sess.run(optimizer,feed_dict={x:x_batch[j],y_:y_batch[j],prob:0.9})
#        sumnum +=1
    sumnum +=1
    accu1=acurracy.eval(feed_dict={x:xs,y_:ys,prob:1})
    accu2=acurracy.eval(feed_dict={x:x_test,y_:y_test,prob:1})      
    if sumnum%25 ==0:
        print(accu1,accu2)
    
print('Accuracy in all data:%f\tAccuracy in test batch:%f'%(accu1,accu2))    
        
electron=np.linspace(-0.9,0.9,101)
pressure=np.linspace(-1,8,101)
electron_mesh,pressure_mesh=np.meshgrid(electron,pressure)
phase=[]

for i in range(len(pressure)):
    phase.append([])
    for j in range(len(electron)):
#        print(i*len(pressure)+j)
        phase[i].append(phase_get.eval(feed_dict={x:[[electron[j],pressure[i]]],
             y_:[[0,0,0,0,0]],prob:1}))


#plt.contourf(pressure_mesh,electron_mesh,phase)
plt.contourf(pressure_mesh,electron_mesh,phase, 10, alpha = 0.6, cmap = plt.cm.Blues)        
        
