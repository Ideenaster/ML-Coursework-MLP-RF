import numpy as np
from Decision_Tree import DecisionTreeClassifier
import time

class RandomForestClassifier:
    """
    简单实现的随机森林分类器, 使用决策树作为基分类器, 采用自助采样法训练每个基分类器
    Args:
        n_estimators(int, optional (default=100)):
            决策树的数量
        max_depth(int, optional (default=None)):
            决策树的最大深度,默认为None
        min_samples_split(int, optional (default=2)):
            节点分裂所需的最小样本数
        max_features:
            取样时的最大特征数
    """
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, max_features=None):
        """
        初始化随机森林分类器

        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.ylen = None
        self.trees = []
    def fit(self, X, y):
        """
        Fit the model with X and y
        Args:
            X: numpy array of shape (n_samples, n_features)
            y: numpy array of shape (n_samples,)
        Returns:
            None

        """
        self.trees = []
        self.ylen = len(np.unique(y))                                 #y的类别数
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split)
            indices = np.random.choice(len(X), len(X), replace=True)  # Bootstrapping
            X_sample, y_sample = X[indices], y[indices]

            if self.max_features:
                feature_indices = np.random.choice(X.shape[1], self.max_features, replace=True)
                # 仅使用部分特征进行训练,采用有放回抽取,也可采用无放回抽取

                X_sample = X_sample[:, feature_indices]
                tree.feature_indices = feature_indices
                #此处将索引存放在树的属性中，以便预测时使用相同的特征子集
                #其实应该也可以将特征子集存放在随机森林的属性中

            start = time.time()
            tree.fit(X_sample, y_sample)
            print("Time:", time.time() - start, "Tree:", _ + 1, "complete", "Depth:", tree.max_depth)
            #打印一些信息

            self.trees.append(tree)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], self.ylen))
        for tree in self.trees:
            tree_pred = tree.predict(X)
            for i, val in enumerate(tree_pred):
                predictions[i, val] += 1
        return np.argmax(predictions, axis=1)

    def predict_proba(self, X):
        """
        输出多分类情况下的软分类概率
        Args:
            X(numpy array of shape (n_samples, n_features)):
                测试数据
        Returns:
            numpy array of shape (n_samples, n_classes):
                在多分类情况下的软分类概率
        """
        predictions = np.zeros((X.shape[0], self.ylen))
        for tree in self.trees:
            tree_pred = tree.predict(X)
            for i, val in enumerate(tree_pred):
                predictions[i, val] += 1
        predictions /= self.n_estimators
        return predictions

if __name__ == '__main__':
    #使用sklearn的iris数据集进行基准测试
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=100, max_depth=3)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Predictions:", y_pred)
    print("True:", y_test)
    print("Proba:", clf.predict_proba(X_test))
