import re
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False



def evalu_turnover(position_df):
    daily_change = position_df.diff().abs()
    daily_turnover = daily_change.sum(axis=1)
    average_turnover = daily_turnover.mean()
    return average_turnover

def cal_single_factor_topBottom_rtn_wcost(df: pd.DataFrame, df_R: pd.DataFrame, up=90, low=10):
    """
    Calculate the single factor_rank return with the transaction cost included. df is the dataframe
    containing the single factor_rank. df_R is the dataframe containing the target return, i.e., the
    ratio of the T+2 vwap price and the T+1 vwap price. The transaction cost is one thousandth for
    one side, i.e., for buy and sell, we calculate the transaction cost only once. We return a
    pandas series of each day's net profit and loss.
    """

    def retain_extreme_10_percent(row):
        row -= np.min(row)  # 确保打分>0
        sorted_row = np.sort(row.dropna())  # 排序并忽略NaN值
        if len(sorted_row) == 0:
            return row  # 如果行全是NaN，则不做处理直接返回

        low_threshold = np.percentile(sorted_row, low)  # 下10%的阈值
        high_threshold = np.percentile(sorted_row, up)  # 上10%的阈值

        new_row = pd.Series(np.nan, index=row.index)

        # 计算权重并赋值
        new_row[row <= low_threshold] = 1 / len(row[row <= low_threshold]) * -0.5
        new_row[row >= high_threshold] = 1 / len(row[row >= high_threshold]) * 0.5

        # 将不在前后10%数值区间内的值设为NaN
        return new_row

    df_F_n = df.apply(retain_extreme_10_percent, axis=1)
    df_F_n = df_F_n.fillna(0)
    df_R_check = df_R.reindex(columns=df.columns)
    df_R_check = df_R_check.loc[df_F_n.index]
    res = (df_F_n * df_R_check).sum(axis=1)
    nav = 1 + res
    nav = nav.cumprod()
    return res, nav, df_F_n


def cal_factor_top_extra_rtn(df: pd.DataFrame, df_R: pd.DataFrame, up=90, pool_df=None):
    '''用中证全指计算多头超额，取因子top10%'''
    def retain_extreme_10_percent(row):
        row -= np.min(row)  # 确保打分>0
        sorted_row = np.sort(row.dropna())  # 排序并忽略NaN值
        if len(sorted_row) == 0:
            return row  # 如果行全是NaN，则不做处理直接返回

        high_threshold = np.percentile(sorted_row, up)  # 上10%的阈值
        new_row = pd.Series(np.nan, index=row.index)
        # 计算权重并赋值
        new_row[row >= high_threshold] = 1 / len(row[row >= high_threshold])  # 等权持有
        # 将不在前后10%数值区间内的值设为NaN
        return new_row

    df_F_n = df.apply(retain_extreme_10_percent, axis=1)
    df_F_n = df_F_n.fillna(0)
    df_R_check = df_R.reindex(columns=df.columns)
    df_R_check = df_R_check.loc[df_F_n.index]
    res = (df_F_n * df_R_check).sum(axis=1)
    if pool_df is not None:
        base_rtn = (df_R * pool_df).mean(axis=1)
    else:
        base_rtn = df_R.mean(axis=1)
    extra_rtn = res - base_rtn
    nav_df = pd.DataFrame()
    nav_df['extra_nav'] = (1 + extra_rtn).cumprod()
    final_extra_rtn = nav_df['extra_nav'].values[-1] -1
    annual_extra_rtn = (1 + final_extra_rtn) ** (252 / len(nav_df)) - 1
    long_turnover = evalu_turnover(df_F_n)
    return {'nav':nav_df, 'ann_rtn':annual_extra_rtn, 'turnover':long_turnover}


def plot_factor_group_return_v2(df_F_smooth_n: pd.DataFrame, df_R: pd.DataFrame, group_num: int):
    """
    Plot the annualized return for each group if we divide all stocks into group_num groups.
    This version is faster than the original version.
    """

    def AnnualizedReturnFun(Ret, N):
        AR = (1 + Ret) ** (252 / N) - 1
        return AR

    df_R_check = df_R.reindex(columns=df_F_smooth_n.columns)
    df_R_check = df_R_check.loc[df_F_smooth_n.index]
    df_F_vals = df_F_smooth_n.values
    df_R_vals = df_R_check.fillna(0).values
    # df_R_vals = df_R.values
    N = df_F_smooth_n.shape[0]
    res = np.empty((N, group_num))
    res[:] = np.nan
    for i in range(N):
        cur_F = df_F_vals[i, :]
        cur_R = df_R_vals[i, :]
        cur_F_R = np.vstack([cur_F, cur_R])
        cur_F_R = cur_F_R[:, ~np.isnan(cur_F_R[0, :])]
        cur_F_R = cur_F_R[:, (-cur_F_R[0, :]).argsort()]
        cur_N = cur_F_R.shape[1]
        stk_num = cur_N // group_num
        temp = []
        for which_group in range(1, group_num + 1):
            B = cur_F_R[1, ((which_group - 1) * stk_num):(which_group * stk_num)]
            C = np.nanmean(B)
            temp.append(C)
        res[i, :] = temp
    df = pd.DataFrame(res)
    cum_rtn = (1 + df.fillna(0)).cumprod().iloc[-1] - 1
    ann_rtn = AnnualizedReturnFun(cum_rtn, N)

    return ann_rtn


# 前复权的vwap
def download_allA_bar_pre_volume_vwap(st_date, end_date):
    stocks = rq.all_instruments(type='CS')
    stocks = stocks['order_book_id'].to_list()
    bar = rq.get_price(stocks, st_date, end_date, '1d', fields=['volume', 'total_turnover'], adjust_type='pre_volume')
    bar['vwap'] = bar['total_turnover'] / bar['volume']
    bar.reset_index(inplace=True)
    bar.rename(columns={'date': 'day_date'}, inplace=True)
    bar['day_date'] = bar['day_date'].dt.strftime('%Y-%m-%d')
    bar['order_book_id'] = rq.id_convert(bar['order_book_id'].to_list(), to='normal')
    wide_bar = bar[['day_date', 'order_book_id', 'vwap']].pivot(index='day_date', columns='order_book_id', values='vwap')
    return wide_bar


def download_allA_bar(st_date, end_date):
    stocks = rq.all_instruments(type='CS')
    stocks = stocks['order_book_id'].to_list()
    bar = rq.get_price(stocks, st_date, end_date, fields=['open','close','high','low','volume','total_turnover'], adjust_type='pre')
    bar.reset_index(inplace=True)
    bar['order_book_id'] = rq.id_convert(bar['order_book_id'].to_list(),to='normal')
    bar.rename(columns={'date': 'datetime'}, inplace=True)
    bar['datetime'] = bar['datetime'].dt.strftime('%Y-%m-%d')
    return


def is_limit_up_down(st_date, end_date):
    '''判断涨跌停'''
    stocks = rq.all_instruments(type='CS')
    stocks = stocks['order_book_id'].to_list()
    price_data = rq.get_price(stocks, start_date=st_date, end_date=end_date, fields=['close', 'limit_up', 'limit_down'])
    # 判断涨停或跌停
    price_data['status'] = 0
    price_data.loc[price_data['close'] == price_data['limit_up'], 'status'] = 1  # 涨停
    price_data.loc[price_data['close'] == price_data['limit_down'], 'status'] = -1  # 跌停
    price_data = price_data.reset_index()

    # 构建结果 DataFrame
    result_df = price_data.pivot(index='date', columns='order_book_id', values='status')

    # 将结果转换为 0/1 格式（涨停或跌停为 1，否则为 0）
    result_df = result_df.abs().fillna(0)
    result_df = result_df.replace({1: np.nan, 0: 1})
    result_df.columns = rq.id_convert(result_df.columns.to_list(), to='normal')
    result_df.index = result_df.index.strftime('%Y-%m-%d')
    return result_df

def down_st_from_rq(st_date, end_date):
    print(f'下载ST')
    stock_df = rq.all_instruments(type='CS')
    stock_list = stock_df['order_book_id'].to_list()
    st_stock_df = rq.is_st_stock(stock_list, st_date, end_date)
    sorted_columns = sorted(st_stock_df.columns)
    st_stock_df = st_stock_df[sorted_columns]
    new_columns = st_stock_df.columns.str.replace('.XSHE', '.SZ').str.replace('.XSHG', '.SH')
    st_stock_df.columns = new_columns
    st_stock_df.index = st_stock_df.index.strftime('%Y-%m-%d')
    st_stock_df = st_stock_df.replace({True: np.nan, False: 1})
    return st_stock_df

# 读取/保存文件参数
params = {
        'dim': 64,  # Transformer嵌入维度
        'depth': 6,  # Transformer层数
        'heads': 8,  # 注意力头数
        'mlp_dim': 256,  # MLP隐藏层维度
        'dropout': 0.1,
        'emb_dropout': 0.1,
        # 训练参数
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
}
p_cls = f"_dim{params['dim']}_mlp{params['mlp_dim']}_depth{params['depth']}_heads{params['heads']}"
p_rate = f"_lr{params['learning_rate']}_dr{params['dropout']}_emb{params['emb_dropout']}_dc{params['weight_decay']}"
# 保存推理结果
params['infer_path_gz'] = f'Vit_classification{p_cls}/{p_rate}.csv.gz'
params['factor_path'] = f'Vit_classification{p_cls}/vision{p_rate}.png'

if __name__ == '__main__':
    factor_name = params['factor_path']
    factor_wide = pd.read_csv(params['infer_path_gz'], index_col=0)
    factor_wide.index = pd.to_datetime(factor_wide.index)

    target_rtn1 = pd.read_csv(r'/home/intern007/dataset/songan_space/input_rq/back_test/allA_T2_T1_vwap_rq_20100101_20251130.csv.xz', index_col=0)
    target_rtn1.rename(columns=lambda x: x[:6], inplace=True)
    target_rtn1.index = pd.to_datetime(target_rtn1.index)
    target_rtn1 = target_rtn1.reindex_like(factor_wide)
    bar_available = pd.read_csv(r'/home/intern007/dataset/songan_space/input_rq/back_test/allA_available_2005-01-01_2025-11-30.csv.xz', index_col=0)
    bar_available.rename(columns=lambda x: x[:6], inplace=True)
    bar_available.index = pd.to_datetime(bar_available.index)
    bar_available = bar_available.reindex_like(factor_wide)
    non_st = pd.read_csv(r'/home/intern007/dataset/songan_space/input_rq/back_test/非ST股票_20100101_20251130.csv.xz', index_col=0)
    non_st.rename(columns=lambda x: x[:6], inplace=True)
    non_st.index = pd.to_datetime(non_st.index)
    non_st = non_st.reindex_like(factor_wide)
    limit_df = pd.read_csv(r'/home/intern007/dataset/songan_space/input_rq/back_test/剔除涨跌停股票_20150101_20251130.csv.gz', index_col=0)
    limit_df.rename(columns=lambda x: x[:6], inplace=True)
    limit_df.index = pd.to_datetime(limit_df.index)
    limit_df = limit_df.reindex_like(factor_wide)


    nonst_factor = non_st * factor_wide.copy()
    nonst_factor = limit_df * nonst_factor.copy()
    nonst_factor = bar_available * nonst_factor.copy()

    rankic_nonst_1 = nonst_factor.corrwith(target_rtn1, method='spearman', axis=1)
    nonst_factor = np.sign(rankic_nonst_1.mean()) * nonst_factor
    _, nav_nonst, nonst_pos_df = cal_single_factor_topBottom_rtn_wcost(nonst_factor.copy(), target_rtn1)
    annual_longshort_rtn = nav_nonst[-1] ** (252 / len(nav_nonst)) - 1
    nonst_turnover = evalu_turnover(nonst_pos_df)
    group_nonst_rtn1 = plot_factor_group_return_v2(nonst_factor.copy(), target_rtn1, 10)
    nonst_long = cal_factor_top_extra_rtn(nonst_factor.copy(), target_rtn1, pool_df=None, up=90)

    #######################多空净值和分组画图####################################
    fig1, axs = plt.subplots(nrows=2, ncols=2, figsize=(19, 10))

    fig1.suptitle(f'{factor_name}, non ST')

    nav_nonst.plot(ax=axs[0][0], title=f'LS ann rtn = {annual_longshort_rtn:.3f}, doubel turnover={round(nonst_turnover, 3)}')

    group_nonst_rtn1.plot(kind='bar', ax=axs[0][1], title='group ann rtn')

    nonst_long['nav'].plot(ax=axs[1][0], title=f"annual_long_extra_rtn={nonst_long['ann_rtn']:.3f}, doubel turnover={nonst_long['turnover']:.3f}, base=equal pool")

    rankic_nonst_1.cumsum().plot(ax=axs[1][1], title=f"rankic_1d={rankic_nonst_1.mean():.3f}")

    plt.tight_layout()
    plt.savefig(factor_name, format='png')
    plt.close()





