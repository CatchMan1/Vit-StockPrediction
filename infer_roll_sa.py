from model import CNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import h5py
from collections import OrderedDict


params_roll = [
    {
        'data_dir': '../images/',  # 替换为实际数据目录
        'trade_dates': '../input/trade_dates.csv',
        'model_dir': './model_result2020/model2020_epoch_16.pth',
        'infer_start': '2021-02-01',
        'infer_end': '2022-01-31',
    },
    {
        'data_dir': '../images/',  # 替换为实际数据目录
        'trade_dates': '../input/trade_dates.csv',
        'model_dir': './model_result2021/model2021_epoch_16.pth',
        'infer_start': '2022-02-01',
        'infer_end': '2023-01-31',
    },
    {
        'data_dir': '../images/',  # 替换为实际数据目录
        'trade_dates': '../input/trade_dates.csv',
        'model_dir': './model_result2022/model2022_epoch_14.pth',
        'infer_start': '2023-02-01',
        'infer_end': '2024-01-31',
    },
    {
        'data_dir': '../images/',  # 替换为实际数据目录
        'trade_dates': '../input/trade_dates.csv',
        'model_dir': './model_result2023/model2023_epoch_11.pth',
        'infer_start': '2024-02-01',
        'infer_end': '2024-08-30',
    },
]

if __name__ == '__main__':
    model = CNN(num_classes=3)
    # 检查是否有可用的 GPU
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)  # 使用多个GPU
    model.to(device)
    factor_list = []
    for params in params_roll:
        ############## 加载保存的模型并处理前缀
        model_state_dict = torch.load(params['model_dir'], map_location=device)

        new_state_dict = OrderedDict()
        for k, v in model_state_dict.items():
            if k.startswith("module."):
                name = k[7:]  # 去掉 'module.'
            else:
                name = k
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
        #################################

        # model.load_state_dict(torch.load(params['model_dir'], map_location=device))
        model.eval()
        trade_dates = pd.read_csv(params['trade_dates'])
        trade_dates = trade_dates[(trade_dates['date']>=params['infer_start']) & (trade_dates['date']<=params['infer_end'])]

        batch_size = 128  # 推理每个批次的大小
        for day in tqdm(trade_dates['date'], desc='CNN infer'):
            outputs_list = []
            # print(day)
            h5_file_path = params['data_dir'] + day +'.h5'
            with h5py.File(h5_file_path, 'r') as h5_file:
                keys = list(h5_file.keys())
                all_images_tensor = torch.stack([
                    torch.from_numpy(h5_file[key][:]).float() for key in keys[:]
                ])
                all_images_tensor = all_images_tensor.to(device)
            with torch.no_grad():
                num_batches = (len(all_images_tensor) + batch_size - 1) // batch_size
                for i in range(num_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(all_images_tensor))
                    batch_tensor = all_images_tensor[start_idx:end_idx].to(device)
                    outputs_batch = model(batch_tensor).float()
                    prob = F.softmax(outputs_batch, dim=1)
                    outputs_list.append(prob.cpu())
                all_prob = torch.cat(outputs_list, dim=0)
                factor = all_prob[:, 2] - all_prob[:, 0]
                data_list = factor.tolist()
                day_df = pd.DataFrame([data_list], columns=keys[:])
                day_df['date'] = day
                factor_list.append(day_df)
            # day_df.to_csv(f'./factor_result/cnn_{day}.csv')
    factor_df = pd.concat(factor_list)
    factor_df.set_index('date', inplace=True)
    factor_df.to_csv(f"cnn_factor_14_rollDiskRankClass_16_16_14_11_{params_roll[0]['infer_start']}_{params_roll[-1]['infer_end']}.csv")



