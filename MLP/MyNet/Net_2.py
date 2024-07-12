import numpy as np
import pandas as pd
import time

"""
    - 函数说明：
    1. l_relu: leaky relu激活函数
    2. softmax: softmax激活函数
    3. sigmoid: sigmoid激活函数
    4. ReLU: ReLU激活函数
       - 含有D_的函数为激活函数对输入的偏导数
"""
def l_relu(z, alpha=0.01):
    return np.maximum(alpha*z, z)

def D_l_relu(z, alpha=0.01):
    return np.where(z > 0, 1, alpha)
def softmax(z):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z, axis=0)

def D_softmax(z):
    s = softmax(z)
    return s * (1 - s)
def sigmoid(z):
    # Limit z to avoid overflow
    z = np.clip(z, -709, 709)  # np.exp(709) is the largest number that doesn't overflow
    return 1 / (1 + np.exp(-z))
def D_sigmoid(z):
    return sigmoid(z) * (1 - sigmoid(z))
def ReLU(z):
    return np.maximum(0, z)
def D_ReLU(z):
    return np.where(z > 0, 1, 0)



class Network(object):
    """
        简单实现的多层感知机
    """
    def __init__(self,layers,layers_activate):
        """
            初始化网络
            Args:
                layers: 各层的节点数
                layers_activate: 各层的激活函数
            Members:
                learning_rate: 学习率
                num_layers: 网络层数
                layers: 各层的节点数
                layers_activate: 各层的激活函数
                biases: 各层的偏置
                weights: 各层的权重
                m_b: Adam优化器的参数
                v_b: Adam优化器的参数
                m_w: Adam优化器的参数
                v_w: Adam优化器的参数
                t: Adam优化器的参数
        """
        self.learning_rate = None
        self.num_layers = len(layers)
        self.layers = layers[1:]  # layers为列表，包含各层的网络节点数
        self.layers_activate = layers_activate
        # 随机初始化权重和偏置
        rand = np.random.RandomState(int(time.time()))
        self.biases = [rand.randn(y) for y in layers[1:]]
        self.weights = [rand.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]
        # 初始化Adam优化器的参数
        self.m_b = [np.zeros(b.shape) for b in self.biases]
        self.v_b = [np.zeros(b.shape) for b in self.biases]
        self.m_w = [np.zeros(w.shape) for w in self.weights]
        self.v_w = [np.zeros(w.shape) for w in self.weights]
        self.t = 0

    def D_cost(self, a, y):
        """
            交叉熵损失函数对输出的偏导数
            Args:
                a: 模型输出
                y: 标签
            Returns:
                交叉熵损失函数对输出的偏导数
        """
        epsilon = 1e-7  # 防止除以零
        return -y / (a + epsilon) + (1 - y) / (1 - a + epsilon)

    def D_fuc(self, z, idx):
        """
        激活函数对输入的偏导数
        Args:
            z: 输入
            idx: 层索引
        Returns:
            激活函数对输入的偏导数
        """
        if self.layers_activate[idx] == sigmoid:
            return D_sigmoid(z)
        if self.layers_activate[idx] == ReLU:
            return D_ReLU(z)
        if self.layers_activate[idx] == softmax:
            return D_softmax(z)
        if self.layers_activate[idx] == l_relu:
            return D_l_relu(z)
    def Forward_prop(self, a):  ##前向传播
        for b, w, fuc in zip(self.biases, self.weights, self.layers_activate):
            a = fuc(np.dot(w, a) + b)
        return a

    def Backward_prop(self, X, y):
        """
            反向传播,根据类中的layers_activate和layers计算梯度
            Args:
                X: 输入
                y: 标签
            Returns:
                grad_b: 偏置的梯度
                grad_w: 权重的梯度
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        # 前向传播
        activate = X
        activate_list = [X]
        z_list = []
        for b, w, fuc in zip(self.biases, self.weights, self.layers_activate):
            z = np.dot(w, activate) + b
            activate = fuc(z)
            z_list.append(z)
            activate_list.append(activate)  # 进行前向传播，记录反向传播所需要的中间变量
        # 反向传播
        delta = self.D_cost(activate_list[-1], y) * self.D_fuc(z_list[-1], -1)
        grad_b[-1] = delta
        grad_w[-1] = np.dot(delta.reshape(self.layers[-1], 1), activate_list[-2].reshape(1, len(activate_list[-2])))
        for i in range(2, self.num_layers):
            z = z_list[-i]
            delta = np.dot(self.weights[-i + 1].T, delta) * self.D_fuc(z, -i)
            grad_b[-i] = delta
            grad_w[-i] = np.dot(delta.reshape(self.layers[-i], 1),
                                activate_list[-i - 1].reshape(1, len(activate_list[-i - 1])))
        return grad_b, grad_w

    # def update(self,batch): ##更新权重和偏置
    #     grad_b = [np.zeros(b.shape) for b in self.biases]
    #     grad_w = [np.zeros(w.shape) for w in self.weights]
    #     for X,y in batch:
    #         delta_grad_b,delta_grad_w = self.Backward_prop(X,y)
    #         grad_w = [gw+dw for gw,dw in zip(grad_w,delta_grad_w)]
    #         grad_b = [gb+dg for gb,dg in zip(grad_b,delta_grad_b)]
    #     self.weights = [w-self.learning_rate/len(batch)*gw for w,gw in zip(self.weights,grad_w)]
    #     self.biases = [b-self.learning_rate/len(batch)*gb for b,gb in zip(self.biases,grad_b)]
    def update(self,batch):
        """
            使用Adam优化器更新权重和偏置
            Args:
                batch: 批量数据

        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]
        for X,y in batch:
            delta_grad_b,delta_grad_w = self.Backward_prop(X,y)
            grad_w = [gw+dw for gw,dw in zip(grad_w,delta_grad_w)]
            grad_b = [gb+dg for gb,dg in zip(grad_b,delta_grad_b)]
        # 使用Adam优化器更新权重和偏置
        self.t += 1
        for i in range(len(self.biases)):
            self.m_b[i] = 0.9 * self.m_b[i] + 0.1 * grad_b[i]
            self.v_b[i] = 0.999 * self.v_b[i] + 0.001 * np.square(grad_b[i])
            m_hat = self.m_b[i] / (1 - 0.9 ** self.t)
            v_hat = self.v_b[i] / (1 - 0.999 ** self.t)
            self.biases[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
        for i in range(len(self.weights)):
            self.m_w[i] = 0.9 * self.m_w[i] + 0.1 * grad_w[i]
            self.v_w[i] = 0.999 * self.v_w[i] + 0.001 * np.square(grad_w[i])
            m_hat = self.m_w[i] / (1 - 0.9 ** self.t)
            v_hat = self.v_w[i] / (1 - 0.999 ** self.t)
            self.weights[i] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)
    def train(self,train_X,train_y,epoch,batch_size,learning_rate):
        ##采用随机梯度下降法训练网络
        self.learning_rate = learning_rate
        train_data = list(zip(train_X,train_y))
        #history用于记录每个epoch的准确率和损失
        history = []
        for i in range(epoch):
            np.random.shuffle(train_data)
            batches = [train_data[k:k+batch_size] for k in range(0,len(train_data),batch_size)]
            for batch in batches:
                self.update(batch)
            acc, precision, recall, f1, cost = self.evaluate(train_data)
            print('Epoch {0}: {1}'.format(i,acc))
            #记录每个epoch的评估参数
            history.append(self.evaluate(train_data))
        return history

    # def evaluate(self,test_data):
    #     ##评估网络的性能
    #     ##模型输出的为正类的概率，取概率大于0.5的为正类，小于的为负类
    #     result = [((self.Forward_prop(x) > 0.5).astype(int),y) for x,y in test_data]
    #     return sum(int(x==y) for x,y in result)
    def evaluate(self, test_data):
        TP = 0
        FP = 0
        FN = 0
        TN = 0
        cost = 0
        # 模型输出的为正类的概率，取概率大于0.5的为正类，小于的为负类
        for x, y in test_data:
            prediction = (self.Forward_prop(x) > 0.5).astype(int)
            if prediction == 1 and y == 1:
                TP += 1
            elif prediction == 0 and y == 1:
                FN += 1
            elif prediction == 1 and y == 0:
                FP += 1
            else:
                TN += 1
        #计算当前的cost
        for x, y in test_data:
            a = self.Forward_prop(x)
            cost += np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
        #计算acc，precision，recall，f1
        acc = (TP + TN) / (TP + FP + FN + TN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        return acc, precision, recall, f1, cost


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






