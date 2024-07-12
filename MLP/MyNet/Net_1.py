import numpy as np
import pandas as pd
import time
def sigmoid(z):
    # Limit z to avoid overflow
    z = np.clip(z, -709, 709)  # np.exp(709) is the largest number that doesn't overflow
    return 1 / (1 + np.exp(-z))
def D_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))
class Network(object):
    def __init__(self,layers,activate_fuc=sigmoid):
        self.learning_rate = None
        self.num_layers = len(layers)
        self.layers = layers[1:]    #layers为列表，包含各层的网络节点数
        #随机初始化权重和偏置
        rand = np.random.RandomState(int(time.time()))
        self.biases = [rand.randn(y) for y in layers[1:]]
        self.weights = [rand.randn(y,x) for x,y in zip(layers[:-1],layers[1:])]
        self.fuc = activate_fuc

    def D_cost(self,a,y):  ##损失函数对输出的偏导数
    ##dcost/da
        a_c = a.copy()
        if self.fuc == sigmoid:
            a_c[y]-=1
        return a_c
    def D_fuc(self,z):  ##激活函数对输入的偏导数
        if self.fuc == sigmoid:
            return D_sigmoid(z)
    def Forward_prop(self,a):  ##前向传播
        for b,w in zip(self.biases,self.weights):
            a = self.fuc(np.dot(w,a)+b)
        return a
    def Backward_prop(self,X,y):  ##反向传播
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        #前向传播
        activate = X
        activate_list = [X]
        z_list = []
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activate)+b
            activate = self.fuc(z)
            z_list.append(z)
            activate_list.append(activate)#进行前向传播，记录反向传播所需要的中间变量
        #反向传播
        delta = self.D_cost(activate_list[-1],y)*D_sigmoid(z_list[-1])
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta.reshape(self.layers[-1],1),activate_list[-2].reshape(1,len(activate_list[-2])))
        for i in range(2,self.num_layers):
            z = z_list[-i]
            delta = np.dot(self.weights[-i+1].T,delta)*self.D_fuc(z)
            grad_b[-i] = delta
            grad_w[-i] = np.dot(delta.reshape(self.layers[-i],1),activate_list[-i-1].reshape(1,len(activate_list[-i-1])))
        return grad_b,grad_w
    def update(self,batch): ##更新权重和偏置
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for X,y in batch:
            delta_grad_b,delta_grad_w = self.Backward_prop(X,y)
            grad_w = [gw+dw for gw,dw in zip(grad_w,delta_grad_w)]
            grad_b = [gb+dg for gb,dg in zip(grad_b,delta_grad_b)]
        self.weights = [w-self.learning_rate/len(batch)*gw for w,gw in zip(self.weights,grad_w)]
        self.biases = [b-self.learning_rate/len(batch)*gb for b,gb in zip(self.biases,grad_b)]

    def train(self,train_X,train_y,epoch,batch_size,learning_rate):
        ##采用随机梯度下降法训练网络
        self.learning_rate = learning_rate
        train_data = list(zip(train_X,train_y))
        for i in range(epoch):
            np.random.shuffle(train_data)
            batches = [train_data[k:k+batch_size] for k in range(0,len(train_data),batch_size)]
            for batch in batches:
                self.update(batch)
            print('Epoch {0}: {1} / {2}'.format(i,self.evaluate(train_data),len(train_data)))

    def evaluate(self,test_data):
        ##评估网络的性能
        result = [(np.argmax(self.Forward_prop(x)),y) for x,y in test_data]
        return sum(int(x==y) for x,y in result)

if __name__ == '__main__':
    #加载数据
    data = pd.read_csv('data/train.csv')
    target = data['label']
    feature = data.drop('label', axis=1)
    #对feature进行归一化处理
    feature = feature/255
    # 去掉表头
    feature = feature.values
    target = target.values
    # sklearn随机切分数据集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.2, random_state=0)
    net = Network([784, 30, 10])
    net.train(X_train, y_train, 20, 10, 1)
    # 评估网络性能
    print(net.evaluate(list(zip(X_test, y_test)))/len(y_test))






