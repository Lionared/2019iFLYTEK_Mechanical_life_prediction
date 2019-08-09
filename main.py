# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from multiprocessing import Pool
from sklearn.model_selection import KFold
from tqdm import tqdm

import warnings

warnings.filterwarnings('ignore')


# 获取数据文件地址
def getfilelist(dir, filelist):
    newdir = dir
    if os.path.isfile(dir):
        filelist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newdir = os.path.join(dir, s)
            getfilelist(newdir, filelist)
    return filelist


# 修改工作时长小于0的值（暂定成nan之后删除）
def trans_to_nan(hours):
    if hours < 0:
        hours = np.nan
    return hours


# 数据处理，去除部件工作时长为负数的值,并且每个时间点只保留k个值
def preprocess(path, k):
    raw_data = pd.read_csv(path, engine='python')

    # 将部件工作时长<0的时长值改为nan并删除
    # print(raw_data.columns)

    raw_data['部件工作时长'] = raw_data['部件工作时长'].map(lambda r: trans_to_nan(r))
    raw_data = raw_data.dropna()

    # 每个工作时长至多保留k个值
    # 提取部件工作时长列作为list方便处理
    raw_list = raw_data['部件工作时长'].tolist()
    for i in range(len(raw_list) - k):
        counter = 1
        # 找到重复项最后一项的索引
        while counter + i < len(raw_list):
            if raw_list[i] == raw_list[i + counter]:
                counter += 1
            else:
                break
        # 判断是否需要删除数据
        if counter <= k:
            continue
        else:
            for m in range(counter - k):
                raw_list[i + k + m] = np.nan
    # 修改dataframe对应列
    raw_data['部件工作时长'] = raw_list
    raw_data = raw_data.dropna()

    return raw_data


# 处理单个单本的数据,添加单个样本的特征
def feature_project(data, df, name, k):
    # 根据样本选择或处理特征

    # 开关与告警信号取其在总数据中的占比
    if name == '开关1信号' or name == '开关2信号' or name == '告警信号1':
        df[name + '时间占比'] = data.sum() / len(data)

    # 温度信号取其均值与标准差为特征
    elif name == '温度信号' or name == '流量信号':
        df[name + '均值'] = data.mean()
        df[name + '标准差'] = data.std()

    # 累积量参数取最大值，k个周期的差分的均值与标准差作为特征
    elif name == '累积量参数1' or name == '累积量参数2':
        df[name] = data.max()
        data = data.diff(periods=k)
        data = data.dropna()
        df[name + str(k) + '阶差分均值'] = data.mean()
        df[name + str(k) + '阶差分标准差'] = data.std()

    # 电流信号主要集中分布在三段区间中，分别列出取均值与标准差，加权后取为特征
    elif name == '电流信号':
        length = len(data)
        low_current = list(num for num in data if 0 <= num < 20)
        mid_current = list(num for num in data if 500 <= num < 750)
        high_current = list(num for num in data if 800 <= num < 1800)
        low_percentage = np.sum(low_current) / length
        mid_percentage = np.sum(mid_current) / length
        high_percentage = np.sum(high_current) / length
        df[name + '低电流段均值'] = np.mean(low_current) * low_percentage
        df[name + '中电流段均值'] = np.mean(mid_current) * mid_percentage
        df[name + '高电流段均值'] = np.mean(high_current) * high_percentage
        df[name + '低电流段标准差'] = np.std(low_current) * low_percentage
        df[name + '中电流段标准差'] = np.std(mid_current) * mid_percentage
        df[name + '高电流段标准差'] = np.std(high_current) * high_percentage

    # 流量信号主要集中分布在三段区间中，分别列出取均值与标准差，加权后取为特征
    elif name == '流量信号':
        length = len(data)
        low_current = list(num for num in data if 0 <= num < 9)
        mid_current = list(num for num in data if 10 <= num < 120)
        high_current = list(num for num in data if 125 <= num < 145)
        low_percentage = np.sum(low_current) / length
        mid_percentage = np.sum(mid_current) / length
        high_percentage = np.sum(high_current) / length
        df[name + '低流量段均值'] = np.mean(low_current) * low_percentage
        df[name + '中流量段均值'] = np.mean(mid_current) * mid_percentage
        df[name + '高流量段均值'] = np.mean(high_current) * high_percentage
        df[name + '低流量段标准差'] = np.std(low_current) * low_percentage
        df[name + '中流量段标准差'] = np.std(mid_current) * mid_percentage
        df[name + '高流量段标准差'] = np.std(high_current) * high_percentage

    # 压力信号1主要分布在两段区间上，同上取均值与标准差加权后取为特征
    elif name == '压力信号1':
        length = len(data)
        low_pressure = list(num for num in data if 65 <= num <= 75)
        high_pressure = list(num for num in data if 180 <= num <= 400)
        low_percentage = np.sum(low_pressure) / length
        high_percentage = np.sum(high_pressure) / length
        df[name + '信号1低压力段标准差'] = np.std(low_pressure) * low_percentage
        df[name + '信号1高压力段标准差'] = np.std(high_pressure) * high_percentage

    # 压力信号2主要分布在一段区间上，剩余值较小，处理同上
    elif name == '压力信号2':
        length = len(data)
        low_pressure = list(num for num in data if 0 <= num <= 50)
        high_pressure = list(num for num in data if 200 <= num)
        low_percentage = np.sum(low_pressure) / length
        high_percentage = np.sum(high_pressure) / length
        df[name + '信号2低压力段标准差'] = np.std(low_pressure) * low_percentage
        df[name + '信号2高压力段标准差'] = np.std(high_pressure) * high_percentage

    # 同压力信号2
    elif name == '转速信号1':
        length = len(data)
        low_pressure = list(num for num in data if 0 <= num <= 100)
        high_pressure = list(num for num in data if 3000 <= num)
        low_percentage = np.sum(low_pressure) / length
        high_percentage = np.sum(high_pressure) / length
        df[name + '信号1低转速段均值'] = np.mean(low_pressure) * low_percentage
        df[name + '信号1高转速段均值'] = np.mean(high_pressure) * high_percentage
        df[name + '信号1低转速段标准差'] = np.std(low_pressure) * low_percentage
        df[name + '信号1高转速段标准差'] = np.std(high_pressure) * high_percentage

    # 同压力信号2
    elif name == '转速信号2':
        length = len(data)
        low_pressure = list(num for num in data if 0 <= num <= 1000)
        high_pressure = list(num for num in data if 10000 <= num)
        low_percentage = np.sum(low_pressure) / length
        high_percentage = np.sum(high_pressure) / length
        df[name + '信号2极低转速段均值'] = np.mean(low_pressure) * low_percentage
        df[name + '信号2高转速段均值'] = np.mean(high_pressure) * high_percentage
        df[name + '信号2极低转速段标准差'] = np.std(low_pressure) * low_percentage
        df[name + '信号2高转速段标准差'] = np.std(high_pressure) * high_percentage

    return df


# 耦合特征构造
def coupled_feature(dataframe, df):
    # 取出列名表
    column_list = dataframe.columns.values.tolist()
    # 循环将特征两两相乘组合
    for i in range(3, 13):
        for j in range(i + 1, 13):
            #            mutiple = dataframe.iloc[:,[i]]*dataframe.iloc[:,[j]]
            mutiple = dataframe.iloc[:, [i]].values * dataframe.iloc[:, [j]].values
            df[column_list[i] + '与' + column_list[j] + '乘积的均值'] = mutiple.mean()
            df[column_list[i] + '与' + column_list[j] + '乘积的标准差'] = mutiple.std()

    return df


# 处理单个训练样本
def process_sample_single(path, train_percentage=1, k=6):
    # 获取并预处理数据
    data = preprocess(path, k)
    # 获取该零件寿命
    work_life = data['部件工作时长'].max()
    # 获取在寿命一定百分比时间的数据
    data = data[data['部件工作时长'] <= work_life * train_percentage]
    # 创建数据集
    dict_data = {'train_file_name': os.path.basename(path) + str(train_percentage),
                 'device': data['设备类型'][0],
                 '开关1_sum': data['开关1信号'].sum(),
                 '开关2_sum': data['开关2信号'].sum(),
                 '告警1_sum': data['告警信号1'].sum(),
                 'current_life': np.log(data['部件工作时长'].max() + 1),
                 'rest_life': np.log(work_life - data['部件工作时长'].max() + 1)
                 }

    # 单项特征
    for item in ['部件工作时长',
                 '累积量参数1',
                 '累积量参数2',
                 '转速信号1',
                 '转速信号2',
                 '压力信号1',
                 '压力信号2',
                 '温度信号',
                 '流量信号',
                 '电流信号',
                 '开关1信号',
                 '开关2信号',
                 '告警信号1']:
        dict_data = feature_project(data[item], dict_data, item, k)

        # 耦合特征
    dict_data = coupled_feature(data, dict_data)

    features = pd.DataFrame(dict_data, index=[0])
    return features


# 多进程调用单文件处理函数，并整合到一起
def get_together(cpu, listp, istest, func):
    if istest:
        train_p_list = [1]
        rst = []
        pool = Pool(cpu)
        for e in listp:
            for train_p in train_p_list:
                rst.append(pool.apply_async(func, args=(e, train_p,)))
        pool.close()
        pool.join()

        # print(rst[0])

        rst = [i.get() for i in rst]

        # print(rst[0])

        tv_features = rst[0]
        for i in rst[1:]:
            tv_features = pd.concat([tv_features, i], axis=0)
        cols = tv_features.columns.tolist()

        try:
            for col in [idx, ycol]:
                cols.remove(col)
            cols = [idx] + cols + [ycol]
        except:
            cols = [idx] + cols + [ycol]

        tv_features[idx] = tv_features[idx].apply(lambda x: x[:-1])
        tv_features = tv_features.reindex(columns=cols)
    else:
        train_p_list = np.arange(0.01, 1, 0.02)  # [0.45,0.55,0.63,0.75,0.85]  #=list(np.arange(0.05,1,0.05))
        rst = []
        pool = Pool(cpu)
        for e in listp:
            for train_p in train_p_list:
                # print train_p
                rst.append(pool.apply_async(func, args=(e, train_p,)))
        pool.close()
        pool.join()
        # print(rst)

        f_list = []

        for i in tqdm(rst):
            f_list.append(i.get())
        # rst = [i.get() for i in tqdm(rst)]
        rst = f_list

        tv_features = rst[0]
        for i in rst[1:]:
            tv_features = pd.concat([tv_features, i], axis=0)
        cols = tv_features.columns.tolist()

        try:
            for col in [idx, ycol]:
                cols.remove(col)
            cols = [idx] + cols + [ycol]
        except:
            cols = [idx] + cols + [ycol]

        tv_features = tv_features.reindex(columns=cols)

    return tv_features


# 评价指标
def compute_loss(target, predict):
    temp = np.log(abs(target + 1)) - np.log(abs(predict + 1))
    res = np.sqrt(np.dot(temp, temp) / len(temp))
    return res


# lgb
def lgb_cv(train, params, fit_params, feature_names, nfold, seed, test):
    train_pred = pd.DataFrame({
        'true': train[ycol],
        'pred': np.zeros(len(train))})
    test_pred = pd.DataFrame({idx: test[idx], ycol: np.zeros(len(test))}, columns=[idx, ycol])
    kfolder = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    for fold_id, (trn_idx, val_idx) in enumerate(kfolder.split(train)):
        print('\nFold_{fold_id} Training ================================\n'.format(fold_id=fold_id))
        lgb_trn = lgb.Dataset(
            data=train.iloc[trn_idx][feature_names],
            label=train.iloc[trn_idx][ycol],
            feature_name=feature_names)
        lgb_val = lgb.Dataset(
            data=train.iloc[val_idx][feature_names],
            label=train.iloc[val_idx][ycol],
            feature_name=feature_names)
        lgb_reg = lgb.train(params=params, train_set=lgb_trn,
                            num_boost_round=fit_params['num_boost_round'], verbose_eval=fit_params['verbose_eval'],
                            early_stopping_rounds=fit_params['early_stopping_rounds'], valid_sets=[lgb_trn, lgb_val])
        val_pred = lgb_reg.predict(
            train.iloc[val_idx][feature_names],
            num_iteration=lgb_reg.best_iteration)

        train_pred.loc[val_idx, 'pred'] = val_pred
        test_pred[ycol] += (np.exp(lgb_reg.predict(test[feature_names])) - 1)
    test_pred[ycol] = test_pred[ycol] / nfold
    score = compute_loss(pd.Series(np.exp(train_pred['true']) - 1).apply(max, args=(0,))
                         , pd.Series(np.exp(train_pred['pred']) - 1).apply(max, args=(0,)))
    print('\nCV LOSS:', score)
    return test_pred


# from tqdm import tqdm
# import time
# for i in tqdm(range(1,100,1)):
#     time.sleep(1)
#     print i
idx = 'train_file_name'
ycol = 'rest_life'

# ====== lgb ======
params_lgb = {'num_leaves': 250,
              'max_depth': 5,
              'learning_rate': 0.01,
              'objective': 'regression',
              'boosting': 'gbdt',
              'verbosity': -1}

fit_params_lgb = {'num_boost_round': 800,
                  'verbose_eval': 200,
                  'early_stopping_rounds': 30}


def get_device_type(path):
    _df = pd.read_csv(path)
    if not _df.empty:
        return _df['设备类型'][0]
    return None


# 执行主进程
if __name__ == '__main__':

    start = time.time()

    train_list = getfilelist('train', [])
    test_list = getfilelist('test1', [])

    n = 4
    func = process_sample_single
    train = get_together(n, train_list, False, func)
    test = get_together(n, test_list, True, func)
    print("done.", time.time() - start)

    train.to_csv('train_total_features.csv', index=False)
    test.to_csv('test_total_features.csv', index=False)
    train.head()
    test.head()
