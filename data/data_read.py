import dolphindb as ddb
import pandas as pd
import numpy as np
import time
import h5py

def query_features_from_dolphin(day_date, vol, buyVol, sellVol, megaBuyVol_largeBuyVol, megaSellVol_largeSellVol, smallBuyVol, smallSellVol, close):
    s = ddb.session()
    s.connect("10.80.65.148", 8848, 'quant_read', 'quant_123456')
    script = f"""
    tab = loadTable('dfs://lv2_feature_HFD_factor', 'minute_feature_trade_entrust')
    factor = select TradeTime, SecurityID, ({vol}), ({buyVol}), ({sellVol}), ({megaBuyVol_largeBuyVol}) as megaBuyVol_largeBuyVol, ({megaSellVol_largeSellVol}) as megaSellVol_largeSellVol, ({smallBuyVol}), ({smallSellVol}), ({close}) from tab where date(TradeTime)={day_date} order by SecurityID
    factor
    """
    factor_df = s.run(script)
    s.close()
    return factor_df

def clean_data(df, vol_cols, price_col='close'):
    """清洗数据：成交量NaN填充0，价格NaN删除行"""
    original_count = len(df)
    original_securities = df['SecurityID'].nunique()
    
    df_clean = df.copy()
    
    # 成交量列的 NaN 填充为 0
    for vol_col in vol_cols:
        df_clean[vol_col] = df_clean[vol_col].fillna(0)
    
    # 删除价格列或 SecurityID 为 NaN 的行
    df_clean = df_clean.dropna(subset=[price_col, 'SecurityID'])
    
    cleaned_count = len(df_clean)
    cleaned_securities = df_clean['SecurityID'].nunique()
    removed_count = original_count - cleaned_count
    removed_securities = original_securities - cleaned_securities
    
    stats = {
        'original_count': original_count,
        'cleaned_count': cleaned_count,
        'removed_count': removed_count,
        'removed_percentage': (removed_count / original_count * 100) if original_count > 0 else 0,
        'original_securities': original_securities,
        'cleaned_securities': cleaned_securities,
        'removed_securities': removed_securities
    }
    
    return df_clean, stats

def create_quantile_array(df_clean, vol_cols, price_col='close', n_bins=8):
    """
    使用 groupby 避免重复过滤
    """
    quant_dict = {}
    n_metrics = len(vol_cols)

    # 使用 groupby 一次性分组，避免重复过滤
    for security_id, stock_data in df_clean.groupby('SecurityID', sort=False):
        tensor_3d = np.zeros((n_metrics, n_bins , n_bins), dtype=int)
        
        if len(stock_data) == 0:
            continue
        
        # 提取价格数据（避免重复访问 DataFrame）
        price_values = stock_data[price_col].values
        
        # 为该股票计算价格分位数边界
        price_quantiles = np.quantile(price_values, np.linspace(0, 1, n_bins + 1))
        price_bins = np.digitize(price_values, price_quantiles[1:-1])
        
        # 为该股票的每个成交量指标生成矩阵
        for metric_idx, vol_col in enumerate(vol_cols):
            # 提取成交量数据
            vol_values = stock_data[vol_col].values
            
            # 为该股票计算该指标的分位数边界
            vol_quantiles = np.quantile(vol_values, np.linspace(0, 1, n_bins + 1))
            vol_bins = np.digitize(vol_values, vol_quantiles[1:-1])
            
            # 向量化统计
            flat_indices = vol_bins * n_bins + price_bins
            counts = np.bincount(flat_indices, minlength=n_bins * n_bins)
            # tensor_4d[stock_idx, metric_idx] = counts.reshape(n_bins, n_bins)
            tensor_3d[metric_idx] = counts.reshape(n_bins, n_bins)

        quant_dict[security_id] = tensor_3d

    return quant_dict

if __name__ == '__main__':
    t0 = time.time()
    # 获取数据
    df1 = query_features_from_dolphin('2017.01.04', 'vol', 'buyVol', 'sellVol', 'megaBuyVol+largeBuyVol', 'megaSellVol+largeSellVol', 'smallBuyVol', 'smallSellVol', 'close')
    # 定义成交量列
    vol_cols = ['vol', 'buyVol', 'sellVol', 'megaBuyVol_largeBuyVol', 'megaSellVol_largeSellVol', 'smallBuyVol', 'smallSellVol']
    # 清洗数据
    df_clean, stats = clean_data(df1, vol_cols=vol_cols, price_col='close')
    # 生成矩阵
    result = create_quantile_array(df_clean, vol_cols=vol_cols, price_col='close', n_bins=8)
    t1 = time.time()
    print(f"数据获取耗时: {t1 - t0:.2f} 秒")
    h5_p = 'data/h5_p.h5'
    with h5py.File(h5_p, 'a') as h5f:
        for stock_code, image in result.items():
            # print(stock_code)
            if stock_code not in h5f:
                h5f.create_dataset(stock_code, data=image, dtype=np.uint8, compression='gzip')