import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

def calculate_factor_rankic(factor1_path, factor2_path, output_plot_path=None):
    factor1 = pd.read_csv(factor1_path, index_col=0)
    factor1.index = pd.to_datetime(factor1.index)
    factor2 = pd.read_csv(factor2_path, index_col=0)
    factor2.index = pd.to_datetime(factor2.index)
    
    
    # 计算每日的 Spearman 秩相关系数
    rankic_series = factor1.corrwith(factor2, method='spearman', axis=1)
    
    # 统计信息
    rankic_mean = rankic_series.mean()
    rankic_std = rankic_series.std()
    rankic_ir = rankic_mean / rankic_std if rankic_std > 0 else 0
    rankic_positive_ratio = (rankic_series > 0).sum() / len(rankic_series)
    
    print(f"\n=== RankIC 统计 ===")
    print(f"平均 RankIC: {rankic_mean:.4f}")
    print(f"RankIC 标准差: {rankic_std:.4f}")
    print(f"RankIC IR (均值/标准差): {rankic_ir:.4f}")
    print(f"RankIC > 0 的比例: {rankic_positive_ratio:.2%}")
    
    # 绘图
    if output_plot_path:
        print(f"\n生成图表: {output_plot_path}")
        fig, axs = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
        
        # 累积 RankIC
        rankic_cumsum = rankic_series.cumsum()
        axs[0].plot(rankic_cumsum.index, rankic_cumsum.values)
        axs[0].set_title(f'cum RankIC (avg_ic={rankic_mean:.4f}, IR={rankic_ir:.4f})')
        axs[0].set_xlabel('date')
        axs[0].set_ylabel('cum RankIC')
        axs[0].grid(True, alpha=0.3)
        
        # 每日 RankIC
        axs[1].plot(rankic_series.index, rankic_series.values, alpha=0.7)
        axs[1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axs[1].axhline(y=rankic_mean, color='g', linestyle='--', alpha=0.5, label=f'avg={rankic_mean:.4f}')
        axs[1].set_title('daily RankIC')
        axs[1].set_xlabel('date')
        axs[1].set_ylabel('RankIC')
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_plot_path, format='png', dpi=150)
        plt.close()
        print(f"图表已保存")
    
    return {
        'rankic_series': rankic_series,
        'rankic_mean': rankic_mean,
        'rankic_std': rankic_std,
        'rankic_ir': rankic_ir,
        'rankic_positive_ratio': rankic_positive_ratio,
        'rankic_cumsum': rankic_series.cumsum()
    }


if __name__ == '__main__':
    # 配置文件路径
    factor1_path = "/mnt/dataset/liziqi_space/vit_classify/results_op1/Vit_classification_dim32_mlp128_depth3_heads4/_lr0.0001_dr0.1_emb0.0_dc0.001.csv.gz" 
    factor2_path = "/mnt/dataset/liziqi_space/vit_classify/results_up/Vit_classification_dim32_mlp128_depth3_heads4/_lr0.0001_dr0.1_emb0.0_dc0.001.csv.gz"
    output_plot_path = 'factor_comparison_rankic.png' 
    
    # 计算 RankIC
    results = calculate_factor_rankic(
        factor1_path=factor1_path,
        factor2_path=factor2_path,
        output_plot_path=output_plot_path
    )
    
    # 保存详细结果
    results_df = pd.DataFrame({
        'date': results['rankic_series'].index,
        'rankic': results['rankic_series'].values,
        'rankic_cumsum': results['rankic_cumsum'].values
    })
    results_df.to_csv('factor_comparison_rankic_details.csv', index=False)
