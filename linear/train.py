# -- coding: utf-8 --
import numpy as np
import os

def getTrains(root):
  X = np.zeros((60000,784)) #初始化数据矩阵
  Y = np.zeros(60000, dtype='uint8') #初始化标签矩阵
  dirs=os.listdir(root)#列出目录路径
  count=0#读取的数据量
  for i in dirs:  #遍历目录中的文件夹
    files=os.listdir(root+'/'+i)#
    for file in files:  #遍历文件夹下的数据
      png=mpimg.imread(root+'/'+i+'/'+file)
      X[count]=png.flatten()  #将图片数据一维化这里即为将二维的图片转为1*784
      Y[count]=int(i) #将文件夹名作为1标签信息存入y矩阵
      count=count+1 #累计读取量
    print(root+'/'+i+' have been done.')  #每读取一个文件夹输出读取完成。
  return X,Y


def getTests(root):
  X = np.zeros((60000,784)) #初始化一个数据矩阵
  Y = np.zeros(60000, dtype='uint8') #初始化标签矩阵
  dirs=os.listdir(root)#列出目录路径
  count=0
  for i in dirs:
    files=os.listdir(root+'/'+i)
    for file in files:
      png=mpimg.imread(root+'/'+i+'/'+file)
      X[count]=png.flatten()
      Y[count]=int(i)
      count=count+1
    print(root+'/'+i+' have been done.')
  return X,Y

trainRoot='Mnist/mnist_train'
testRoot='Mnist/mnist_test'

X,Y=getTrains(trainRoot)  #读取路径下的数据与标签
X_,Y_=getTests(testRoot)

W = 0.01 * np.random.randn(784,10)#随机初始化权重W
b = np.zeros((1,10))#偏置体现在能使分界在x移动和在Y上移动，所以这里需要2维度
step_size = 0.0001#设置步长(学习率)
num = X.shape[0]

#w与b的学习迭代1000次
for i in range(1000):
  F = np.dot(X, W) + b   #   f = w*x + b
  S = np.exp(F) / np.sum(np.exp(F), axis=1, keepdims=True)  
  loss = -np.log(S[range(num),Y])  #softmax 互熵损失
  L = np.sum(loss)/num #整体的损失
  dS = S
  dS[range(num),Y] -= 1
  dW = np.dot(X.T, dS)#计算偏导数，梯度
  db = np.sum(dS/num, axis=0, keepdims=True)#偏置的形式决定了它是面向整体的
  #更新W和b
  W=W-step_size*dW
  b=b-step_size*db
  if i%10==0:
  	print(u'目前的损失为 %.2f' % (L))
  	train_accuracy=np.mean(np.argmax(np.dot(X, W) + b, axis=1) ==Y)
  	test_accuracy=np.mean(np.argmax(np.dot(X_, W) + b, axis=1) ==Y_)
  	print(u'训练精度为 %.2f' % (train_accuracy))
  	print(u'测试精度为 %.2f' % (test_accuracy))