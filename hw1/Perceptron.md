

```python
# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
from __future__ import division
```


```python
#加载训练数据
def load_data(path):
    data = pd.read_csv(path, sep='\s+', header=None)
    row, col = data.shape
    data = data.as_matrix()
    X = np.c_[np.ones((row, 1)), data[:, 0: col-1]]
    Y = data[:,col-1:col]
    return X, Y
```


```python
#PLA算法
def pla(X, Y, W, alpha=1):
    count = 0;prepos = 0
    while True:
        y_hat = np.sign(np.dot(X, W))
        y_hat[np.where(y_hat == 0)] = -1
        index = np.where(y_hat!=Y)[0]
        if not index.any():
            break
        if not index[index >= prepos].any():
            prepos = 0
        pos = index[index >= prepos][0]
        W += alpha * Y[pos,0] * X[pos:pos+1,:].T
        count += 1
        prepos = pos
    return W, count
```


```python
#Pocket算法
def pocket(X, Y, W, iterations, alpha=1):
    count = 0
    y_hat = np.sign(np.dot(X, W))
    y_hat[np.where(y_hat == 0)] = -1
    W_pocket = np.zeros(W.shape)
    while count < iterations:
        index = np.where(y_hat!=Y)[0]
        if not index.any():
            break
        pos = index[np.random.permutation(len(index))][0]
        W = W + alpha * Y[pos,0] * X[pos:pos+1,:].T
        y_hat = np.sign(np.dot(X, W))
        y_hat[np.where(y_hat == 0)] = -1
        if len(np.where(y_hat!=Y)[0]) < len(index):
            W_pocket = W
        count += 1
    return W_pocket, W
```


```python
url="https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_15_train.dat"
X, Y = load_data(url)
row, col = X.shape
```


```python
# Q15
W = np.zeros((col,1))
W,count = pla(X, Y, W)
print "pla更新总次数：", count
```

    pla更新总次数： 39



```python
# Q16
total = 0
for i in range(2000):
    W = np.zeros((col,1))
    rand_index = np.random.permutation(row)
    X_rand = X[rand_index]
    Y_rand = Y[rand_index]
    W,count = pla(X_rand, Y_rand, W)
    total += count
print "pla平均运行次数：", total/2000
```

    pla平均运行次数： 40.081



```python
# Q17
total = 0
for i in range(2000):
    W = np.zeros((col,1))
    rand_index = np.random.permutation(row)
    X_rand = X[rand_index]
    Y_rand = Y[rand_index]
    W,count = pla(X_rand, Y_rand, W, 0.5)
    total += count
print "alpha=0.5 pla平均运行次数：", total/2000
```

    alpha=0.5 pla平均运行次数： 39.876



```python
url="https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_18_train.dat"
X_train, Y_train = load_data(url)
row, col = X_train.shape
url="https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_18_test.dat"
X_test, Y_test = load_data(url)
```


```python
# Q18
total = 0
for i in range(2000):
    W = np.zeros((col,1))
    rand_index = np.random.permutation(row)
    X_rand = X_train[rand_index]
    Y_rand = Y_train[rand_index]
    W_pocket,W = pocket(X_rand, Y_rand, W, 50)
    y_hat = np.sign(np.dot(X_test, W_pocket))
    y_hat[np.where(y_hat == 0)] = -1
    error_rate = len(np.where(y_hat!=Y_test)[0])/row
    total += error_rate
print "迭代50次，W_pocket平均错误率：", total/2000
```

    pocket平均错误率： 0.198654



```python
# Q19
total = 0
for i in range(2000):
    W = np.zeros((col,1))
    rand_index = np.random.permutation(row)
    X_rand = X_train[rand_index]
    Y_rand = Y_train[rand_index]
    W_pocket,W = pocket(X_rand, Y_rand, W, 50)
    y_hat = np.sign(np.dot(X_test, W))
    y_hat[np.where(y_hat == 0)] = -1
    error_rate = len(np.where(y_hat!=Y_test)[0])/row
    total += error_rate
print "迭代50次，W平均错误率：", total/2000
```

    迭代50次，W平均错误率： 0.353644



```python
# Q20
total = 0
for i in range(2000):
    W = np.zeros((col,1))
    rand_index = np.random.permutation(row)
    X_rand = X_train[rand_index]
    Y_rand = Y_train[rand_index]
    W_pocket,W = pocket(X_rand, Y_rand, W, 100)
    y_hat = np.sign(np.dot(X_test, W_pocket))
    y_hat[np.where(y_hat == 0)] = -1
    error_rate = len(np.where(y_hat!=Y_test)[0])/row
    total += error_rate
print "迭代50次，W_pocket平均错误率：", total/2000
```

    迭代50次，W平均错误率： 0.192073

