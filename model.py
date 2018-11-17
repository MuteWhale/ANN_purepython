import numpy as  np

class model(object):
    def __init__(self,input_size,n_hidden,output_size,weight,bias):
        self.input_size = input_size
        self.n_hidden = n_hidden
        self.output_size = output_size
        self.bias = bias
        if not weight:
            if bias:
                self.weight=[]
                weight0= np.random.random((self.input_size, self.n_hidden+1))
                self.weight.append(weight0)
                weight1 = np.random.random((self.n_hidden+1, self.n_hidden+1))
                self.weight.append(weight1)
                weight2 = np.random.random((self.n_hidden+1, self.output_size))
                self.weight.append(weight2)
            else:
                self.weight = []
                weight0 = np.random.random((self.input_size, self.n_hidden))
                self.weight.append(weight0)
                weight1 = np.random.random((self.n_hidden , self.n_hidden))
                self.weight.append(weight1)
                weight2 = np.random.random((self.n_hidden , self.output_size))
                self.weight.append(weight2)
        else:
            self.weight=weight

    # x shape is (batch_size,n)
    def relu(self,x):
        return (np.abs(x) + x) / 2.0
        #x的绝对值加自己再除以2

    # y shape is (batch_size,n)
    def diffRelu(self, y):
        temp=np.ones(y.shape)
        return temp*(y>0)
        # (y>0是一个true 和 false 的矩阵~~~哈哈哈，乘出来刚好所有的非正项为0~)

    # x 形状为（batch_size,input_size）,y2 形状为（batch_size,input_size）
    def forward(self,x):
        y0=self.relu(np.matmul(x,self.weight[0]))
        y1=self.relu(np.matmul(y0,self.weight[1]))
        y2=self.relu(np.matmul(y1,self.weight[2]))
        return y2

    def backPropagation(self, x, y):

        #正向传播并且保留每一层的结果~
        ys = []
        y0 = self.relu(np.matmul(x, self.weight[0]))
        ys.append(y0)
        y1 = self.relu(np.matmul(y0, self.weight[1]))
        ys.append(y1)
        y2 = self.relu(np.matmul(y1, self.weight[2]))
        ys.append(y2)

        #反向传播计算斜率 ,由于计算时需复合交替使用对细胞的偏导，和对权值的偏导
        round_cell = []
        round_weight = []
        round_cell.append(self.diffRelu(ys[-1] - y))
        # 反一层
        round_cell[0].shape=(10,1)
        ys[-2].shape=(self.n_hidden,1)
        round_weight.append(np.matmul(ys[-2],round_cell[0].T))
        round_cell.append(self.diffRelu(np.matmul(round_cell[0].T, self.weight[-1].T)))
        # 反二层
        round_cell[1].shape=(self.n_hidden,1)
        ys[-3].shape=(self.n_hidden,1)
        round_weight.append(np.matmul(ys[-3].T,round_cell[1]).T)
        round_cell.append(self.diffRelu(np.matmul(round_cell[1].T, self.weight[-2].T)))
        # 输出
        round_cell[2].shape=(self.n_hidden,1)
        x.shape=(self.input_size,1)
        round_weight.append(np.matmul(x,round_cell[2].T))


        round_weight.reverse()

        return round_weight


    def gradDesent(self,batch_x,batch_y,lr):

        #对一个batch里面的数据得到权值偏导矩阵进行加和，skrskr
        #每一次内存里面只有一batch中的一个（x，y），计算了权值的偏导之后加进total里面
        round_weight_toal = [np.zeros(w.shape) for w in self.weight]
        for x, y in zip(batch_x,batch_y):
            round_weight = self.backPropagation(x, y)
            round_weight_toal = [a+b for a,b in zip(round_weight_toal, round_weight)]


        #更新权值
        self.weight = [w - (lr/len(batch_y))*round
                       for w, round in zip(self.weight, round_weight_toal)]

    def SGD(self, train_x,train_y, batch_size, lr):
        ''' 随机梯度下降： train_data是data_wrapper()包装之后的训练数据，数据格式见data_wrapper()函数定义处； epochs为迭代次数； batch_size为采样时的批量数据的大小； alpha是学习率； cv_data为可选参数，是data_wrapper()包装之后的交叉验证数据，数据格式见data_wrapper()函数定义处； 若给出了交叉验证数据，则在每次训练后都会进行性能评估，可用以跟踪进度，但会拖慢执行速度。 '''
        m = len(train_y)
        #''' 在每个迭代期，首先将数据打乱，然后将它分成多个小批量数据batches； 对于每一个小批量数据batch应用一次梯度下降，通过调用self.grad_desent()完成。 '''
        indices = np.arange(train_y.shape[0])
        np.random.shuffle(indices)
        x = train_x[indices]
        y = train_y[indices]

        print('first epoch is ready to train')

        #每一个batch都更新权重
        loss = 0
        for  k in range(0, m, batch_size):
            self.gradDesent(x[k: k+batch_size],y[k: k+batch_size],lr)
            error = (self.forward(x[k: k+batch_size])-y[k: k+batch_size])
            loss =loss+np.mean(error*error)
            if k%200==0:
                print('acc now is '+str(self.evaluate(x[k: k+batch_size],y[k: k+batch_size])))

        print('the loss of this batch is'+str(loss/m))

#   def evaluate(self, test_data):
#   ''' 评价函数，预测正确的个数。 np.argmax函数返回数组的最大值的序号，实现从one-hot到数字的转换； ''' test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data] return sum(int(x == y) for (x, y) in test_results) def predict(self, x): '''预测函数''' return np.argmax(self.feedforward(x))
#    def saveModel(self):
    def evaluate(self, test_x,test_y):
        ''' 评价函数，预测正确的个数。 np.argmax函数返回数组的最大值的序号，实现从one-hot到数字的转换； '''
        pre_y = self.predict(test_x)
        result = 0
        for y1, y2 in zip(pre_y,test_y):
            if y1 == np.argmax(y2):
                result+=1

        return result/len(pre_y)
              

    def predict(self, x):
        '''预测函数'''
        #最大的
        y=self.forward(x)
        a=[]
        for i in y:
            a.append(np.argmax(i))
        return a





