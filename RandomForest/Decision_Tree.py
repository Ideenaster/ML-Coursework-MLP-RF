import numpy as np

class Node:
    """
    树的节点类
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self):
        return self.value is not None

class DecisionTreeClassifier:
    """
    决策树分类器
    Args:
        max_depth(int, optional (default=None)):
            树的最大深度. 如果为None,则节点会一直扩展直到所有叶子节点都包含相同的标签或者包含小于min_samples_split个样本点

        min_samples_split(int, optional (default=2)):
            节点在分裂之前必须具有的最小样本数
    """

    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.feature_indices = None  # 新增特征子集属性

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        total = sum(counts)
        entropy = 0
        for count in counts:
            p = count / total
            entropy -= p * np.log2(p)
        return entropy

    def gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        total = sum(counts)
        gini = 1
        for count in counts:
            p = count / total
            gini -= p ** 2
        return gini

    def information_gain(self, y, y_left, y_right, criterion='entropy'):
        """
        信息增益计算函数
        Args:
            y(array-like):
                父节点的标签
            y_left(array-like):
                左子节点的标签
            y_right(array-like):
                右子节点的标签
            criterion(str, optional (default='entropy')):
                信息增益的度量标准,可以是'entropy'或者'gini'

        """
        if criterion == 'entropy':
            return self.entropy(y) - (len(y_left) / len(y)) * self.entropy(y_left) - (len(y_right) / len(y)) * self.entropy(y_right)
        if criterion == 'gini':
            return self.gini(y) - (len(y_left) / len(y)) * self.gini(y_left) - (len(y_right) / len(y)) * self.gini(y_right)

    def split(self, X, y, criterion='entropy'):
        """
        寻找最佳分割特征和阈值
        Args:
            X(array-like):
                特征数据
            y(array-like):
                标签
            criterion(str, optional (default='entropy')):
                信息增益的度量标准,可以是'entropy'或者'gini'
        """
        best_gain = 0
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None

        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature] <= threshold)
                right_indices = np.where(X[:, feature] > threshold)
                y_left = y[left_indices]
                y_right = y[right_indices]
                gain = self.information_gain(y, y_left, y_right, criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        # 如果没有可分割的特征,返回None
        if best_gain == 0:
            return None, None, None, None

        return best_feature, best_threshold, best_left_indices, best_right_indices

    def build_tree(self, X, y, depth=0, criterion='entropy'):
        """
        递归构建树
        Args:
            X(array-like):
                特征数据
            y(array-like):
                标签
            depth(int):
                当前深度
            criterion(str, optional (default='entropy')):
                信息增益的度量标准,可以是'entropy'或者'gini'
        """
        if depth == self.max_depth or len(np.unique(y)) == 1 or len(y) < self.min_samples_split:
            unique_classes, counts = np.unique(y, return_counts=True)
            return Node(value=unique_classes[np.argmax(counts)])

        feature, threshold, left_indices, right_indices = self.split(X, y, criterion=criterion)

        # 如果最佳gain为0或者没有可分割的特征,直接创建叶子节点
        # 避免叶子节点中出现无threshold的情况
        if feature is None or threshold is None:
            unique_classes, counts = np.unique(y, return_counts=True)
            return Node(value=unique_classes[np.argmax(counts)])

        left = self.build_tree(X[left_indices], y[left_indices], depth + 1, criterion)
        right = self.build_tree(X[right_indices], y[right_indices], depth + 1, criterion)
        return Node(feature=feature, threshold=threshold, left=left, right=right)

    def fit(self, X, y, criterion='entropy'):
        self.root = self.build_tree(X, y, criterion = criterion)

    def predict(self, X):
        """
        预测
        Args:
            X(array-like):
                特征数据
        Returns:
            预测结果
        """
        if self.feature_indices is not None:
            X = X[:, self.feature_indices]
        predictions = []
        for sample in X:
            node = self.root
            while not node.is_leaf():
                if sample[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(node.value)
        return predictions


if __name__ == "__main__":
    # 导入肿瘤数据

    from sklearn import datasets

    data = datasets.load_breast_cancer()
    X = data.data
    y = data.target
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
    # 训练模型
    clf = DecisionTreeClassifier(max_depth=10)
    #对训练过程进行计时
    import time
    start = time.time()
    clf.fit(X_train, y_train,criterion='gini')
    end = time.time()
    print("训练时间：",end-start)
    # 预测
    y_pred = clf.predict(X_test)
    # 计算准确率
    accuracy = np.sum(y_test == y_pred) / len(y_test)
    print("Accuracy:", accuracy)

