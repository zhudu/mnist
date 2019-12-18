# -- coding: utf-8 --
import numpy as np
import matplotlib.image as mpimg
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