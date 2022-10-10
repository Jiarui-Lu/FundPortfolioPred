import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import shgo
from scipy.optimize import nnls


class constraint_linear_model(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.num_x = np.shape(X)[1]

    def sum_squares(self, x):  ##二乘法定义
        ls = 0.5 * (self.Y - np.dot(self.X, x)) ** 2
        result = np.sum(ls)
        return result

    def is_sum_equal_1(self, x):
        return 1 - np.sum(x)

    def main(self):
        cons = ({'type': 'eq', 'fun': self.is_sum_equal_1})  ##回归系数满足和为1
        bnds = [(0, 1)]  ##确定系数的边界，满足系数均大于0且小于1
        for i in range(self.num_x - 1):
            bnds.append((0, 1))
        reg = shgo(self.sum_squares,
                   bounds=bnds,
                   constraints=cons)

        return reg


class Cluster_Analysis(object):
    def __init__(self, ETF, stock, n_clusters):
        self.ETF = self.PreprocessData(ETF)  # 原始ETF数据
        self.stock = self.PreprocessData(stock)  # 原始成分股股票池数据
        self.new_ETF, self.new_stock = self.merge_index()
        self.n_clusters = n_clusters

    # 原始数据预处理，计算returns
    def PreprocessData(self, old):
        df = pd.DataFrame(index=old.index, columns=old.columns)
        for col in old.columns:
            df[col] = old[col].pct_change()
        return df.dropna()

    # 处理空值项，并合并index列，为后续回归做准备
    def merge_index(self):
        if len(self.ETF) > len(self.stock):
            new_ETF = self.ETF.loc[self.stock.index, :]
            new_stock = self.stock
        else:
            new_stock = self.stock.loc[self.ETF.index, :]
            new_ETF = self.ETF
        return new_ETF, new_stock

    def Cluster(self, data, n_clutsers):
        # K-Means聚类
        km = KMeans(n_clusters=n_clutsers)
        result = km.fit(data)
        labels = result.labels_  # 获取聚类标签
        return labels
        # centroids = result.cluster_centers_  # 获取聚类中心
        # print(centroids)
        # inertia = result.inertia_  # 获取聚类准则的总和
        # print(inertia)

    def ConstructDict(self, labels):
        d = dict()
        for label in labels:
            d[label] = []
        return d

    def ConstructIndex(self, stock_list):
        stock_slicer = self.new_stock.loc[:, stock_list]
        return_mean = stock_slicer.mean(axis=1)
        return pd.DataFrame(return_mean)

    def Standardlized(self, df):
        s = pd.Series(df[df.columns[0]], index=df.index)
        # print(s)
        _sum = sum(s)
        new_s = s / sum(s)
        # print(new_s)
        new_df = pd.DataFrame(new_s)
        return new_df

    def main(self):
        labels = self.Cluster(self.new_stock.T, n_clutsers=self.n_clusters)
        # print(self.new_stock.T)
        # print(labels)
        stock_dict = self.ConstructDict(set(labels))
        for label in labels:
            _index = list(labels).index(label)
            stock_dict[label].append(self.new_stock.columns[_index])
        Cluster_df = pd.DataFrame()
        for v in stock_dict.values():
            tmp_df = self.ConstructIndex(v)
            Cluster_df = pd.concat([Cluster_df, tmp_df], axis=1)
        for ETF_col in self.new_ETF.columns:
            coef, resid = nnls(Cluster_df, self.new_ETF[ETF_col])
            df = pd.DataFrame(list(coef), columns=[ETF_col])
            result_df = self.Standardlized(df)
            print(result_df)


if __name__ == '__main__':
    stock = pd.read_excel('stock_test.xlsx', index_col=0)
    ETF = pd.read_excel('test.xlsx', index_col=0)
    n_clusters = 10
    x = Cluster_Analysis(ETF, stock, n_clusters)
    x.main()
