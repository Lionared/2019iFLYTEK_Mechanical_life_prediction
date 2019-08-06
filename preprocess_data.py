#-*- coding: utf-8 -*-

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from multiprocessing import Pool
from scipy.stats import pearsonr
from sklearn.model_selection import KFold

import warnings
warnings.filterwarnings('ignore')

#获取文件地址
def get_filelist (dir,flielist):

    new_dir = dir
    if os.path.isfile(dir):
        flielist.append(dir)
    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            new_dir = os.path.join(dir,s)
            get_filelist(new_dir,flielist)
    return flielist


#修改工作时长小于0的值（暂定成nan之后删除）
def trans_to_nan (hours):
    if hours <0:
        hours = np.nan
    return hours


#数据处理，去除部件工作时长为负数的值,并且每个时间点只保留k个值
def preprocess (path,k):

    raw_data = pd.read_csv(path)

    #将部件工作时长<0的时长值改为nan并删除
    raw_data['部件工作时长'] = raw_data['部件工作时长'].map(lambda r:trans_to_nan(r))
    raw_data = raw_data.dropna()

    #每个工作时长至多保留k个值
    #提取部件工作时长列作为list方便处理
    raw_list = raw_data['部件工作时长'].tolist()
    for i in range(len(raw_list)-k):
        counter = 1
        #找到重复项最后一项的索引
        while counter + i < len(raw_list):
            if raw_list[i] == raw_list[i+counter]:
                counter += 1
            else:
                break
        #判断是否需要删除数据
        if counter <= k:
            continue
        else:
            for m in range(counter-k):
                raw_list[i+k+m] = np.nan
    #修改dataframe对应列
    raw_data['部件工作时长'] = raw_list
    raw_data = raw_data.dropna()

    return raw_data


#处理单个单本的数据,添加单个样本的特征
def feature_project (data,df,name,k):

    #根据样本选择或处理特征

    #开关与告警信号取其在总数据中的占比
    if name == '开关1信号' or name == '开关2信号' or name == '告警信号1':
        df[name + '时间占比'] = data.sum()/len(data)
    
    #温度信号取其均值与标准差为特征
    elif name == '温度信号' :
        df[name + '均值'] = data.mean()
        df[name + '标准差'] = data.std()
    
    #累积量参数取最大值，k个周期的差分的均值与标准差作为特征
    elif name == '累积量参数1' or name == '累积量参数2':
        df[name] = data.max()
        data = data.diff(periods = k)
        data = data.dropna()
        df[name + str(k) + '阶差分均值'] = data.mean()
        df[name + str(k) + '阶差分标准差'] = data.std()

    #电流信号主要集中分布在三段区间中，分别列出取均值与标准差，加权后取为特征
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

    #流量信号主要集中分布在三段区间中，分别列出取均值与标准差，加权后取为特征
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
    
    #压力信号1主要分布在两段区间上，同上取均值与标准差加权后取为特征
    elif name == '压力信号1':
        length = len(data)
        low_pressure = list(num for num in data if 65 <= num <=75)
        high_pressure = list(num for num in data if 180 <= num <= 400)
        low_percentage = np.sum(low_pressure) / length
        high_percentage = np.sum(high_pressure) / length
        df[name + '信号1低压力段标准差'] = np.std(low_pressure) * low_percentage
        df[name + '信号1高压力段标准差'] = np.std(high_pressure) * high_percentage
    
    #压力信号2主要分布在一段区间上，剩余值较小，处理同上
    elif name == '压力信号2':
        length = len(data)
        low_pressure = list(num for num in data if 0 <= num <=50)
        high_pressure = list(num for num in data if 200 <= num)
        low_percentage = np.sum(low_pressure) / length
        high_percentage = np.sum(high_pressure) / length
        df[name + '信号2低压力段标准差'] = np.std(low_pressure) * low_percentage
        df[name + '信号2高压力段标准差'] = np.std(high_pressure) * high_percentage

    #同压力信号2
    elif name == '转速信号1':
        length = len(data)
        low_pressure = list(num for num in data if 0 <= num <=100)
        high_pressure = list(num for num in data if 3000 <= num)
        low_percentage = np.sum(low_pressure) / length
        high_percentage = np.sum(high_pressure) / length
        df[name + '信号1低转速段均值'] = np.mean(low_pressure) * low_percentage
        df[name + '信号1高转速段均值'] = np.mean(high_pressure) * high_percentage
        df[name + '信号1低转速段标准差'] = np.std(low_pressure) * low_percentage
        df[name + '信号1高转速段标准差'] = np.std(high_pressure) * high_percentage
    
    #同压力信号2
    elif name == '转速信号2':
        length = len(data)
        low_pressure = list(num for num in data if 0 <= num <=1000)
        high_pressure = list(num for num in data if 10000 <= num)
        low_percentage = np.sum(low_pressure) / length
        high_percentage = np.sum(high_pressure) / length
        df[name + '信号2极低转速段均值'] = np.mean(low_pressure) * low_percentage
        df[name + '信号2高转速段均值'] = np.mean(high_pressure) * high_percentage
        df[name + '信号2极低转速段标准差'] = np.std(low_pressure) * low_percentage
        df[name + '信号2高转速段标准差'] = np.std(high_pressure) * high_percentage

    return df


#耦合特征构造
def coupled_feature (dataframe,df):
    
    #取出列名表
    column_list = dataframe.columns.values.tolist()

    #循环将特征两两相乘组合
    for i in range (3,10):
        for j in range (i+1,10):
            mutiple = dataframe.iloc[:,[i]].values*dataframe.iloc[:,[j]].values
            df[column_list[i] +'与'+ column_list[j] +'乘积的均值'] = mutiple.mean()
            df[column_list[i] +'与'+ column_list[j] +'乘积的标准差'] = mutiple.std()
    
    #循环构造平方项特征
    for i in range (3,10):
        square = dataframe.iloc[:,[i]].values * dataframe.iloc[:,[i]].values
        df[column_list[i] + '平方项的均值'] = square.mean()
        df[column_list[i] + '平方项的标准差'] = square.std()

    return df


#处理单个训练样本
def process_single_sample (path,train_percentage):

    #获取并预处理数据
    data = preprocess(path,12)
    #获取该零件寿命
    work_life = data['部件工作时长'].max()
    #获取在寿命一定百分比时间的数据
    data=data[data['部件工作时长']<=work_life*train_percentage]
    #创建数据集
    dict_data = { 'train_file_name': os.path.basename(path) + str(train_percentage),
                        'device': data['设备类型'][0],
                        '开关1_sum':data['开关1信号'].sum(),
                        '开关2_sum':data['开关2信号'].sum(),
                        '告警1_sum':data['告警信号1'].sum(),
                        'current_life':np.log(data['部件工作时长'].max()+1),
                        'rest_life':np.log(work_life-data['部件工作时长'].max()+1)
                     }

    #单项特征
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
        dict_data=feature_project(data[item],dict_data,item,12)

    #耦合特征
    dict_data=coupled_feature(data,dict_data)

    features = pd.DataFrame(dict_data, index=[0])
    return features


#整合处理训练集与测试集,并采用多线程
def integrated_process (cpu,path_list,test_or_not,func):
    
    #测试集处理
    if test_or_not == True:
        #测试集无需对数据进行分割处理
        train_percentage_list = [1]
        feature_list = []
        rst = []
        pool = Pool(cpu)
        for path in path_list:
            for train_percentage in train_percentage_list:
                rst.append(pool.apply_async(func, args=(path,train_percentage)))
        pool.close()
        pool.join()
        rst = [i.get() for i in rst]
        feature_list=rst[0]
        for item in rst[1:]:
            feature_list = pd.concat([feature_list, item], axis=0)
        columns=feature_list.columns.tolist()
        for col in ['train_file_name','rest_life']:
            columns.remove(col)
        columns=['train_file_name']+columns+['rest_life']
        feature_list['train_file_name']=feature_list['train_file_name'].apply(lambda x:x[:-1])
        feature_list=feature_list.reindex(columns=columns)

    #训练集处理
    if test_or_not == False:
        #训练集目的为预测剩余寿命，故将数据集分割
        train_percentage_list = [0.45,0.55,0.63,0.75,0.85]
        feature_list = []
        rst = []
        pool = Pool(cpu)
        for path in path_list:
            for train_percentage in train_percentage_list:
                rst.append(pool.apply_async(func, args=(path,train_percentage)))
        pool.close()
        pool.join()
        rst = [i.get() for i in rst]
        feature_list=rst[0]
        for item in rst[1:]:
            feature_list = pd.concat([feature_list, item], axis=0)
        columns=feature_list.columns.tolist()
        for col in ['train_file_name','rest_life']:
            columns.remove(col)
        columns=['train_file_name']+columns+['rest_life']
        feature_list=feature_list.reindex(columns=columns)
    
    return feature_list

#评价指标
def compute_loss(target, predict):
    temp = np.log(abs(target + 1)) - np.log(abs(predict + 1))
    res = np.sqrt(np.dot(temp, temp) / len(temp))
    return res

#pearson系数
def pearson(train):

    column_num_list = []

    for i in range(2,train.shape[1]-1):
        length = len(train['rest_life'])
        a = train.iloc[:,[i]].values.flatten()
        b = train['rest_life'].values.flatten()
        r,p = pearsonr(a,b) 
        if r == np.nan:
            column_num_list.append(i)
    
    train.drop(train.columns[column_num_list],axis=1,inplace=True)
    return train


#lgb
def lgb_cv(train, params, fit_params,feature_names, nfold, seed,test):
    train_pred = pd.DataFrame({
        'true': train['rest_life'],
        'pred': np.zeros(len(train))})
    test_pred = pd.DataFrame({'train_file_name': test['train_file_name'], 'rest_life': np.zeros(len(test))},columns=['train_file_name','rest_life'])
    kfolder = KFold(n_splits=nfold, shuffle=True, random_state=seed)
    for fold_id, (trn_idx, val_idx) in enumerate(kfolder.split(train)):
        print('\nFold_{fold_id} Training ================================\n'.format(fold_id = fold_id))
        lgb_trn = lgb.Dataset(
            data=train.iloc[trn_idx][feature_names],
            label=train.iloc[trn_idx]['rest_life'],
            feature_name=feature_names)
        lgb_val = lgb.Dataset(
            data=train.iloc[val_idx][feature_names],
            label=train.iloc[val_idx]['rest_life'],
            feature_name=feature_names)
        lgb_reg = lgb.train(params=params, train_set=lgb_trn,
                            num_boost_round = fit_params['num_boost_round'], verbose_eval = fit_params['verbose_eval'],
                            early_stopping_rounds = fit_params['early_stopping_rounds'], #**fit_params,\
                  valid_sets=[lgb_trn, lgb_val])
        val_pred = lgb_reg.predict(
            train.iloc[val_idx][feature_names],
            num_iteration=lgb_reg.best_iteration)
        
        train_pred.loc[val_idx, 'pred'] = val_pred
        test_pred['rest_life'] += (np.exp(lgb_reg.predict(test[feature_names]))-1) 
    test_pred['rest_life'] = test_pred['rest_life'] / nfold
    score = compute_loss(pd.Series(np.exp(train_pred['true']) - 1).apply(max, args=(0,))
                         ,pd.Series(np.exp(train_pred['pred']) - 1).apply(max, args=(0,)))
    print('\nCV LOSS:', score)
    return test_pred


# ====== lgb ======
params_lgb = {'num_leaves': 250, 
              'max_depth':5, 
              'learning_rate': 0.01,
              'objective': 'regression', 
              'boosting': 'gbdt',
              'verbosity': -1}

fit_params_lgb = {'num_boost_round': 800, 
                  'verbose_eval':200,
                  'early_stopping_rounds': 200}

#主进程，调试使用
if __name__ == '__main__':

    #获取路径集
    start = time.time()
    train_path = 'train'
    test_path = 'test1'
    n=4

    train_list = get_filelist(train_path,[])
    test_list = get_filelist(test_path,[])


    func=process_single_sample
    train=integrated_process(n,train_list,False,func)
    test =integrated_process(n,test_list,True,func)
    print("done.", time.time()-start)

    train_test=pd.concat([train,test],join='outer',axis=0).reset_index(drop=True)
    train_test=pd.get_dummies(train_test,columns=['device'])

    # sub= lgb_cv(train_test.iloc[:train.shape[0]] ,params_lgb, fit_params_lgb, 
    #             feature_name, 5,2018,train_test.iloc[train.shape[0]:])

    # sub.to_csv('baseline_sub1.csv',index=False)
    # print("process(es) done.", time.time()-start)
    
    # data = process_single_sample(train_list[0],1,12)
    # for i in range(data.shape[1]):
    #     print (data.iloc[:,[i]])
    nfold = 3
    seed = 2018

    column_names = train_test.columns.values.tolist()
    special_column_names = ['device_S100','device_S26a','device_S508','device_S51d','device_Saa3','开关1_sum','开关2_sum','告警1_sum']
    special_column_names = ['train_file_name'] + ['current_life'] + special_column_names + ['rest_life']

    for item in special_column_names:
        column_names.remove(item)
        
    train_test.fillna(0,inplace=True)
        
    for item in column_names:
        std_temp = train_test[item].std()
        
        if std_temp <= 1:
            train_test[item] = np.exp(train_test[item])
            std_temp2 = train_test[item].std()
            
            #check the standard deviation again
            if std_temp2 < 1:
                del train_test[item]
            
        elif std_temp > 10:
            train_test[item] = np.log(train_test[item] + 1)

    train_test = pearson(train_test)
    feature_name=list(filter(lambda x:x not in['train_file_name','rest_life'],train_test.columns))

    sub= lgb_cv(train_test.iloc[:train.shape[0]] ,params_lgb, fit_params_lgb, 
                feature_name, nfold,seed,train_test.iloc[train.shape[0]:])

    sub.to_csv('baseline_sub1.csv',index=False)
    print("process(es) done.", time.time()-start)

    
##----------------------这里是用于测试的函数----------------------##

    ##不同type的元件个数
    # list_type = ['S51d', 'S26a', 'S100', 'Saa3', 'S508']
    # num_type = [0,0,0,0,0]
    # num_type_2 = [0,0,0,0,0]
    # for train_path in train_list:
    #     data = pd.read_csv(train_path)
    #     for i in range(5):
    #         if data['设备类型'][0] == list_type[i]:
    #             num_type[i] = num_type[i] + 1
    # for test_path in test_list:
    #     data = pd.read_csv(test_path)
    #     for i in range(5):
    #         if data['设备类型'][0] == list_type[i]:
    #             num_type_2[i] = num_type_2[i] + 1
    # print (num_type)
    # print (num_type_2)

    # #训练集中type分布
    # list_type = ['S51d', 'S26a', 'S100', 'Saa3', 'S508']
    # for i in range(5):
    #     exec('life_list{}=list()'.format(i))
    # for train_path in train_list:
    #     data = pd.read_csv(train_path)
    #     for i in range(5):
    #         if data['设备类型'][0] == list_type[i]:
    #             # max_life = data['部件工作时长'].max()
    #             exec('life_list{}.append(data.部件工作时长.max())'.format(i))
    # for i in range(5):
    #     exec('life_list{}.sort()'.format(i))
    # plt.figure()
    # for i in range(5):
    #     # exec('plt.hist(life_list{})'.format(i))
    #     plt.subplot(231+i)
    #     exec('counter = len(life_list{})+1'.format(i))
    #     exec('plt.scatter(range(1,counter),life_list{})'.format(i))
    #     plt.title(list_type[i],fontsize = 20)
    # plt.show()
    
    # #绘制图像观测数据分布特征
    # plt.figure()
    # for csv_path in train_list:
    #     raw_data = pd.read_csv(csv_path)
    #     plt.scatter(raw_data['部件工作时长'],raw_data['告警信号1'])

    #     #窗口最大化
    #     mng = plt.get_current_fig_manager()
    #     mng.window.state('zoomed') #works fine on Windows!

    #     plt.show()