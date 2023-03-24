import os
import numpy as np
from matplotlib import pyplot as plt
from rich.console import Console
from rich.table import Table
import sys
from pathlib import Path
sys.path.append(str(Path(os.path.abspath(__file__)).parent.parent))
from utils import *

# 感知机
class Perceptron:
    def __init__(self, lr=1e-1, max_iteration=2000, verbose=False):
        self.lr = lr
        self.verbose = verbose
        self.max_iteration = max_iteration

    def _trans(self, x):
        return self.w @ x + self.b

    def _predict(self, x):
        return 1 if self._trans(x) >= 0. else -1

    def fit(self, X, Y):
        # 对于二维张量，shape[0]代表行数，shape[1]代表列数，shape[-1]表示最后一个维度，对于二维数据，此时也是列数
        self.feature_size = X.shape[-1]
        # define parameteres
        self.w = np.random.rand(self.feature_size)
        self.b = np.random.rand(1)

        updated = 1
        epoch = 0
        # if there is mis-classified sample, train
        while updated > 0 and epoch < self.max_iteration:
            # 冗余日志
            if self.verbose:
                print(f"epoch {epoch} started...")

            # 用于统计分类错误的数据个数
            updated = 0
            # shuffle data
            # permutation函数只能针对一维数据随机排列,对于多维数据只能对第一维度的数据进行随机排列,相当于对行进行随机排列
            perm = np.random.permutation(len(X))
            for i in perm:
                x, y = X[i], Y[i]
                # if there is a mis-classified sample
                if self._predict(x) != y:
                    # update the parameters
                    self.w += self.lr * y * x
                    self.b += self.lr * y
                    updated += 1

            if self.verbose:
                print(f"epoch {epoch} finishied, {updated} pieces of data mis-classified")
            epoch += 1
        return

    def predict(self, X):
        # apply_along_axis(func, axis, arr, *args, **kwargs)
        # 将arr数组的每一个元素经过func函数变换形成的一个新数组
        # axis表示函数func对arr是作用于行还是列
        return np.apply_along_axis(self._predict, axis=-1, arr=X)

if __name__ == "__main__":
    def demonstrate(X, Y, desc):
        console = Console(markup=False)
        perceptron = Perceptron(verbose=True)
        perceptron.fit(X, Y)

        # plot
        ''' plt.scatter()
        函数用于生成一个scatter散点图。
        matplotlib.pyplot.scatter(x,
                                  y,
                                  s=20,
                                  c='b',
                                  marker='o',
                                  cmap=None,
                                  norm=None,
                                  vmin=None,
                                  vmax=None,
                                  alpha=None,
                                  linewidths=None,
                                  verts=None,
                                  hold=None,
                                  **kwargs)

        参数：
        x，y：表示的是shape大小为(n, )的数组，也就是我们即将绘制散点图的数据点，输入数据。
        s：表示的是大小，是一个标量或者是一个shape大小为(n, )的数组，可选，默认20。
        c：表示的是色彩或颜色序列，可选，默认蓝色’b’。但是c不应该是一个单一的RGB数字，也不应该是一个RGBA的序列，因为不便区分。c可以是一个RGB或RGBA二维行数组。
        marker：MarkerStyle，表示的是标记的样式，可选，默认’o’。
        cmap：Colormap，标量或者是一个colormap的名字，cmap仅仅当c是一个浮点数数组的时候才使用。如果没有申明就是image.cmap，可选，默认None。
        norm：Normalize，数据亮度在0 - 1之间，也是只有c是一个浮点数的数组的时候才使用。如果没有申明，就是默认None。
        vmin，vmax：标量，当norm存在的时候忽略。用来进行亮度数据的归一化，可选，默认None。
        alpha：标量，0 - 1之间，可选，默认None。
        linewidths：也就是标记点的长度，默认None。
        '''

        # X[:,0]，numpy切片，表示取列号为0的值
        plt.scatter(X[:, 0], X[:, 1], c=Y)
        wbline(perceptron.w, perceptron.b)
        plt.title(desc)
        plt.show()

        # show in table
        pred = perceptron.predict(X)
        table = Table('x', 'y', 'pred')
        for x, y, y_hat in zip(X, Y, pred):
            table.add_row(*map(str, [x, y, y_hat]))
        console.print(table)

    # -------------------------- Example 1 ----------------------------------------
    print("Example 1:")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, 1, -1, -1])
    demonstrate(X, Y, "Example 1")

    # -------------------------- Example 2 ----------------------------------------
    print("Example 2: Perceptron cannot solve a simple XOR problem")
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([1, -1, -1, 1])
    demonstrate(X, Y, "Example 2: Perceptron cannot solve a simple XOR problem")