from WindPy import w
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.optimize import shgo
import numpy as np
from scipy.optimize import nnls

# w.start()
# w.isconnected()


# 导入Wind数据
class ReadWindData(object):
    # 定义Wind函数的初始参数
    def __init__(self, ETF_list, stock_list, start_date, end_date):
        self.ETF_list = ETF_list
        self.stock_list = stock_list
        self.start_date = start_date
        self.end_date = end_date

    # 从Wind数据库中导入沪深300ETF的数据
    def ReadETF(self):
        error_code, df = w.wsd(self.ETF_list, "nav", self.start_date, self.end_date, usedf=True)
        print(df)
        df.to_excel(r'result\test.xlsx')
        return df

    # 从Wind数据库中导入沪深300成分股的收盘价数据
    def ReadStock(self):
        error_code, df = w.wsd(self.stock_list, "close", self.start_date, self.end_date, usedf=True)
        df.to_excel(r'result\stock_test.xlsx')
        return df

    # 主函数
    def main(self):
        ETF = self.ReadETF()
        stock = self.ReadStock()
        return ETF, stock

'''
带约束的线性回归，满足系数为（0，1）且各系数和为1，对多系数回归时时间较慢。
这里主要采用nnls（非负最小二乘法），进行归一化后再令每个系数除以和以满足系数和为1。
'''
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


# Sequential Oscillator Selction
class SOS(object):
    def __init__(self, ETF, stock):
        self.ETF = self.PreprocessData(ETF)  # 原始ETF数据
        self.stock = self.PreprocessData(stock)  # 原始成分股股票池数据
        self.new_ETF, self.new_stock = self.merge_index()
        self.Moveon = True  # 定义循环结束条件

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

    # 线性回归函数，返回均方根误差
    def LR(self, x, y):
        x = pd.DataFrame(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        rms = sqrt(mean_squared_error(y_test, y_pred))
        return rms
        # print(model.coef_)
        # print(model.intercept_)

    # 首次回归，寻找单个股票，单独测试每个因子以找出哪个因子是预测效果最好的。
    def FindMinSingleEquityAndRmse(self, y):
        rms_list = []
        for col in self.new_stock.columns:
            rms_list.append(self.LR(self.new_stock[col], y))
        min_index = rms_list.index(min(rms_list))
        return (self.new_stock.columns[min_index], min(rms_list))
        # print(rms_list)

    # 标准化，满足系数和为1
    def Standardlized(self, df):
        s = pd.Series(df[df.columns[0]], index=df.index)
        # print(s)
        _sum = sum(s)
        new_s = s / sum(s)
        # print(new_s)
        new_df = pd.DataFrame(new_s)
        return new_df

    '''
    主函数，实现震荡优化前向算法通过不停的增加股票这种迭代方式以优化当前规模为N的投资组合，保留RMSE最佳的N + 1
    只股票的投资组合。当前向算法无法进一步降低RMSE时，执行后向算法，
    即通过不停的减少股票这种迭代方式以优化当前规模为N的投资组合，保留RMSE最佳的N - 1只股票的投资组合。
    当后向算法无法进一步降低RMSE时，再次执行前向算法。当两个算法都无法进一步改善RMSE时，便确定了最终的投资组合。
    '''
    def main(self):
        for ETF_col in self.new_ETF.columns:
            self.Moveon = True
            MinSingleEquity, MinRmse = self.FindMinSingleEquityAndRmse(self.new_ETF[ETF_col])
            tmpPortfolio = []
            tmpPortfolio.append(MinSingleEquity)
            while self.Moveon == True:
                tmpPortfolio_pre = tmpPortfolio.copy()
                # print(tmpPortfolio_pre)
                for stock in self.new_stock.columns:
                    if stock != MinSingleEquity:
                        tmpPortfolio.append(stock)
                        tmp_rms = self.LR(self.new_stock.loc[:, tmpPortfolio], self.new_ETF[ETF_col])
                        if tmp_rms < MinRmse:
                            tmp_rms, MinRmse = MinRmse, tmp_rms
                            print(tmpPortfolio)
                        else:
                            tmpPortfolio.remove(stock)
                for stock in tmpPortfolio:
                    if stock != MinSingleEquity:
                        tmpPortfolio.remove(stock)
                        tmp_rms = self.LR(self.new_stock.loc[:, tmpPortfolio], self.new_ETF[ETF_col])
                        if tmp_rms < MinRmse:
                            tmp_rms, MinRmse = MinRmse, tmp_rms
                            print(tmpPortfolio)
                        else:
                            tmpPortfolio.append(stock)
                tmpPortfolio_post = tmpPortfolio.copy()
                # print(tmpPortfolio_pre)
                # print(tmpPortfolio_post)
                if len(tmpPortfolio_pre) == len(tmpPortfolio_post):
                    self.Moveon = False
            coef, resid = nnls(self.new_stock.loc[:, tmpPortfolio], self.new_ETF[ETF_col])
            df = pd.DataFrame(list(coef), index=tmpPortfolio, columns=[ETF_col])
            result_df = self.Standardlized(df)
            result_df.to_excel(r'result\{}_pred_portfolio.xlsx'.format(ETF_col))
        return


if __name__ == '__main__':
    # ETF选择招商沪深300ESG基准ETF、招商沪深300增强ETF、易方达沪深300非银ETF、易方达沪深300医药卫生ETF四只
    # 股票选择沪深300的8月31日成分股
    stock_list = pd.read_excel(r'data\沪深300成分.xlsx', index_col=0).index
    stock_list = ','.join(stock_list)
    ETF_list = "561900.OF,561990.OF,512070.OF,512010.OF"
    start_date = "2021-09-01"
    end_date = "2022-08-31"
    # read = ReadWindData(ETF_list, stock_list, start_date, end_date)
    # ETF, stock = read.main()
    ETF = pd.read_excel(r'data\test.xlsx', index_col=0)
    stock = pd.read_excel(r'data\stock_test.xlsx', index_col=0)
    x = SOS(ETF, stock)
    x.main()
    df=pd.read_excel(r'data\acc.xlsx')
    real=list(df['real1'].dropna())
    pred=list(df['pred1'].dropna())
    acc=[]
    for code in real:
        if code in pred:
            acc.append(1)
        else:
            acc.append(0)
    print(acc)
    print(sum(acc)/len(acc))

