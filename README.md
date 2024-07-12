# 对于多层感知机和随机森林的简单实现

## 模型使用方法

报告正文中所提到的训练完毕的模型使用pickle存储在.pkl文件中，进行验证测试的使用方法如下：

1. 导入模型文件

```python
from MLP.MyNet.Net_2 import Network
```

2. 加载模型

```python
import pickle
with open('model_path.pkl', 'rb') as f:
    model = pickle.load(f)
```

3. 进行预测：

```python
predictions = model.Forward_prop(your_data)
```

ramdomforest模型的使用类似

## 文件夹目录结构

```
.
├── README.md
├── MLP
│   └── MyNet
│       ├── Baseline.ipynb
│       ├── EDA.ipynb
│       ├── Final.ipynb
│       ├── Net_1.py
│       ├── Net_2.py
│       ├── data
│       │   ├── ...
│       ├── data_ex
│       │   ├── ...
│       ├── image
│       │   ├── ...
│       ├── model-train_baseline.pkl
│       ├── model_final_0.86228.pkl
│       ├── result_0.85.csv
│       ├── result_baseline.csv
│       └── result_final(0.86228).csv
└── RandomForest
    ├── Decision_Tree.py
    ├── RandomForest.pkl
    ├── RandomForest.py
    ├── RandomForest_sklearn.pkl
    ├── bank_churn.ipynb
    ├── data
    │   ├── ...
    ├── model_entropy_0.85.pkl
    └── result_0.85.csv
```

- `MLP/MyNet/Net_2.py`:MLP的最终实现版本，Net_1为针对手写数字识别的基准测试版本，没有分层激活函数和adam优化器。
- `MLP/MyNet/EDA.ipynb`: 数据的探索性数据分析。
- `MLP/MyNet/data_ex`: 最终选题的数据存放目录，/data目录为基准测试，也即选题四的数据存放位置
- `MLP/MyNet/result_final(0.86228).csv`: 可向kaggle直接提价的预测结果文件,`_final`或是`_baseline`为此预测结果的模型来源

- `RandomForest/Decision_Tree.py`: 决策树的实现。
- `RandomForest/RandomForest.py`: 随机森林的实现。