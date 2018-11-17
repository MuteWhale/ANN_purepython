import mnist_reader
import numpy as  np
import model

#超参数
lr = 0.001
batch_size = 100
epoch_size = 100




#装载数据
def transform_one_hot(labels):
  n_labels = np.max(labels) + 1
  one_hot = np.eye(n_labels)[labels]
  return one_hot
x_train, y_train = mnist_reader.load_mnist('./data', kind='train')
x_test, y_test = mnist_reader.load_mnist('./data', kind='t10k')
print(len(y_test))
y_train = transform_one_hot(y_train)

#定义模型
ann = model.model(input_size=784,n_hidden=512,output_size=10,weight=None,bias=None)

for i in  range(epoch_size):
    print('----------------------------------epoch %d',i)
    ann.SGD(x_train,y_train,batch_size,lr)
