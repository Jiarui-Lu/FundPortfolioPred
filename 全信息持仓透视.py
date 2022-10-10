import pandas as pd
import numpy as np
from WindPy import *
from scipy.optimize import shgo


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
        # df.to_excel(r'data\test2.xlsx')
        return df

    # 从Wind数据库中导入沪深300成分股的收盘价数据
    def ReadStock(self):
        error_code, df = w.wsd(self.stock_list, "close", self.start_date, self.end_date, usedf=True)
        # df.to_excel(r'result\stock_test.xlsx')
        return df

    # 主函数
    def main(self):
        ETF = self.ReadETF()
        stock = self.ReadStock()
        return ETF, stock


'''
马赛克原理
全信息持仓补全法
Mosaic类：
计算约束条件中的各个矩阵，包括：
SEC为k * n的矩阵，k为证监会行业数目，n为股票池中股票数目，个股对应行业在矩阵中的对应元素为1，否则为0，
TOP_10为n * n的矩阵，若股票池中第i个股票为季报所披露的前十大持仓股，则矩阵元素(i,i)为1，否则为0，矩阵中的非对角线元素为0。
POS_MAX为n * n的矩阵，若股票池中第i个股票为季报所披露的前十大持仓股，则矩阵元素(i,i)为0，否则为1，矩阵中的非对角线元素为0。
POS_MIN为n * n的对角阵。
'''


class Mosaic(object):
    def __init__(self, stock_pool, top_holding):
        self.stock_pool = stock_pool
        self.stock_industry = self.dictconstruct(self.stock_pool['证监会行业'])
        self.top_holding = top_holding
        self.SEC = self.SEC_vector(self.stock_industry, self.stock_pool)
        self.TOP_10 = self.TOP_10_vector(self.stock_pool)
        self.POS_MAX = self.POS_MAX_vector(self.stock_pool)
        self.POS_MIN = self.POS_MIN_vector(self.stock_pool)

    def dictconstruct(self, l):
        new_l = sorted(list(set(l)))
        new_d = dict(enumerate(new_l))
        return dict(zip(new_d.values(), new_d.keys()))

    def SEC_vector(self, k, n):
        sec = np.zeros((len(k), len(n)))
        for stock in n.index:
            ind = n.loc[stock, '证监会行业']
            stock_index = list(n.index).index(stock)
            sec[k[ind], stock_index] = 1
        return sec

    def TOP_10_vector(self, n):
        top = np.zeros((len(n), len(n)))
        for stock in n.index:
            stock_index = list(n.index).index(stock)
            if stock in self.top_holding.keys():
                top[stock_index, stock_index] = 1
        return top

    def POS_MAX_vector(self, n):
        pos_max = np.zeros((len(n), len(n)))
        for stock in n.index:
            stock_index = list(n.index).index(stock)
            if stock not in self.top_holding.keys():
                pos_max[stock_index, stock_index] = 1
        return pos_max

    def POS_MIN_vector(self, n):
        pos_min = np.zeros((len(n), len(n)))
        for stock in n.index:
            stock_index = list(n.index).index(stock)
            pos_min[stock_index, stock_index] = 1
        return pos_min


'''
主要最小值优化函数的类
继承了Mosaic类
读取收益率数据
stock_ret为股票池收益率向量，
stock_ret1、stock_ret2分别为季报发布日前后最近一个交易日股票池的收益率向量。
fund_ret1、fund_ret2为对应交易日该基金的收益率。
并计算相应的n*1阶矩阵，包括：
sec为k维季报披露的证监会行业持仓比例向量。
top_10为n维前十大持仓股持仓比例向量，对应个股为前十大重仓股则元素值为对应权重，否则为0。
pos_max为n维持仓比例阈值向量，对应个股为前十大重仓股则元素值为1，否则为第十大重仓股的持仓比例。
pos_min为元素全部为0的向量。
最终利用shgo函数实现最小值优化
返回持仓股及持仓权重
'''


class Portfolio_Optimize(Mosaic):
    def __init__(self, stock_pool, top_holding,
                 ETF, stock_price, time, industry_percent, pos_tot):
        super(Portfolio_Optimize, self).__init__(stock_pool, top_holding)
        self.time = time
        self.new_ETF = self.ReadQuarterdata(ETF.pct_change(), self.time)
        self.new_stock_price = self.ReadQuarterdata(stock_price.pct_change(), self.time)
        self.industry_percent_list = industry_percent.values()
        self.top_10 = self.cal_top_10(self.stock_pool)
        self.pos_max = self.cal_pos_max(self.stock_pool)
        self.pos_min = self.cal_pos_min(self.stock_pool)
        self.pos_tot = pos_tot

    def _strTofloat(self, l):
        return np.array(list(map(float, l))).reshape(-1, 1)

    def ReadQuarterdata(self, df, time):
        time_list = [time.strftime('%Y-%m-%d') for time in df.index]
        time_index = time_list.index(time)
        # arr=np.array((df.iloc[time_index - 1:time_index + 2, :]))
        # return arr.reshape([-1,3])
        return df.iloc[time_index - 1:time_index + 2, :]

    def cal_top_10(self, n):
        top_10 = np.zeros((len(n), 1))
        for stock in n.index:
            stock_index = list(n.index).index(stock)
            if stock in self.top_holding.keys():
                top_10[stock_index, 0] = self.top_holding[stock]
        return top_10

    def cal_pos_max(self, n):
        pos_max = np.zeros((len(n), 1))
        min_top_10 = min(self.top_holding.values())
        for stock in n.index:
            stock_index = list(n.index).index(stock)
            if stock in self.top_holding.keys():
                pos_max[stock_index, 0] = 1
            else:
                pos_max[stock_index, 0] = min_top_10
        return pos_max

    def cal_pos_min(self, n):
        pos_min = np.zeros((len(n), 1))
        return pos_min

    def sum_squares(self, x):
        stock_ret1 = np.array(self.new_stock_price.iloc[0, :]).reshape(1, -1)
        stock_ret2 = np.array(self.new_stock_price.iloc[-1, :]).reshape(1, -1)
        fund_ret1 = np.array(self.new_ETF.iloc[0, :])
        fund_ret2 = np.array(self.new_ETF.iloc[-1, :])
        ls = float(np.dot(stock_ret1, x) - fund_ret1) ** 2 + (np.dot(stock_ret2, x) - fund_ret2) ** 2
        return ls

    def is_equal_SEC(self, x):
        return np.sum(np.dot(self.SEC, x) - self._strTofloat(self.industry_percent_list))

    def is_equal_top_10(self, x):
        return np.sum(np.dot(self.TOP_10, x) - self._strTofloat(self.top_10))

    def is_smaller_than_posmax(self, x):
        return np.sum(self.pos_max - np.dot(self.POS_MAX, x))

    def is_greater_than_posmin(self, x):
        return np.sum(np.dot(self.POS_MIN, x) - self.pos_min)

    def is_equal_postot(self, x):
        return np.sum(x) - self.pos_tot

    def main(self):
        cons = (
            {'type': 'ineq', 'fun': self.is_smaller_than_posmax},
            {'type': 'ineq', 'fun': self.is_greater_than_posmin},
            {'type': 'eq', 'fun': self.is_equal_SEC},
            {'type': 'eq', 'fun': self.is_equal_top_10},
            {'type': 'eq', 'fun': self.is_equal_postot})
        bnds = [(0, 1), ] * len(self.stock_pool)
        reg = shgo(self.sum_squares, bounds=bnds, constraints=cons)
        return reg


if __name__ == '__main__':
    stock_pool = pd.read_excel(r'data\沪深300成分.xlsx', index_col=0).sort_index()
    stock_list = ','.join(stock_pool.index)
    ETF_list = "510360.OF"
    start_date = "2022-07-01"
    end_date = "2022-07-31"
    top_holding_stock_list = "600519.SH,300750.SZ,600036.SH,601318.SH,601012.SH,000858.SZ,002594.SZ" \
                             ",000333.SZ,601166.SH,300059.SZ".split(',')
    top_holding_percent_list = "0.0595,0.0346,0.0244,0.0236,0.0189,0.0183,0.0141,0.0138,0.0135,0.0124".split(',')
    top_holding = dict(zip(top_holding_stock_list, top_holding_percent_list))
    top_holding = dict(sorted(top_holding.items(), key=lambda x: x[0]))
    read = ReadWindData(ETF_list, stock_list, start_date, end_date)
    ETF, stock_price = read.main()
    # ETF = pd.read_excel(r'data\test2.xlsx', index_col=0)
    # stock_price = pd.read_excel(r'data\stock_test.xlsx', index_col=0)
    # mosaic = Mosaic(stock_pool,top_holding_stock_ list)
    stock_price = (stock_price.T.sort_index()).T
    industry = "农、林、牧、渔业,采矿业,制造业,电力、热力、燃气及水生产和供应业,建筑业,批发和零售业,交通运输、仓储和邮政业" \
               ",信息传输、软件和信息技术服务业,金融业,房地产业,租赁和商务服务业,科学研究和技术服务业" \
               ",教育,卫生和社会工作,文化、体育和娱乐业".split(',')
    percent = "0.019,0.0276,0.5814,0.0246,0.0191,0.0025,0.026,0.0301,0.1974,0.0184,0.0138,0.015,0.0005,0.0086,0.0012".split(
        ',')
    industry_percent = dict(zip(industry, percent))
    industry_percent = dict(sorted(industry_percent.items(), key=lambda x: x[0]))
    '''
        {'交通运输、仓储和邮政业': 0, '信息传输、软件和信息技术服务业': 1, '农、林、牧、渔业': 2, '制造业': 3, 
         '卫生和社会工作': 4, '建筑业': 5, '房地产业': 6, '批发和零售业': 7,
         '教育': 8, '文化、体育和娱乐业': 9, '电力、热力、燃气及水生产和供应业': 10, 
         '科学研究和技术服务业': 11, '租赁和商务服务业': 12, '采矿业': 13, '金融业': 14}
    '''
    pos_tot = 0.9765
    port = Portfolio_Optimize(stock_pool, top_holding, ETF, stock_price, '2022-07-20', industry_percent, pos_tot)
    coef = pd.DataFrame(port.main().x).T
    coef.columns = port.stock_pool.index
    coef.to_excel(r'result\pred_portfolio.xlsx')
    print(coef)
