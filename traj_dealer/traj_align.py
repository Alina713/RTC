# /nas/user/wyh/TNC/data/ETA/diff1_map/SHmap_train.csv
# /nas/user/wyh/TNC/data/ETA/SHmap_train.csv
import pandas as pd


train_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_train.csv", sep=';', header=0)
valid_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_valid.csv", sep=';', header=0)
# test_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_test.csv", sep=';', header=0)

train_trajectory_data = [[int(y) for y in x.strip('[]').split(',')] for x in train_data.iloc[:,1].values]
valid_trajectory_data = [[int(y) for y in x.strip('[]').split(',')] for x in valid_data.iloc[:,1].values]

# 正样本aug
aug_train_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/diff1_map/SHmap_train.csv", sep=';', header=0)
aug_valid_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/diff1_map/SHmap_valid.csv", sep=';', header=0)
# aug_test_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/diff1_map/SHmap_test.csv", sep=';', header=0)

aug_train_trajectory_data = [[int(y) for y in x.strip('[]').split(',')] for x in aug_train_data.iloc[:,1].values]
aug_valid_trajectory_data = [[int(y) for y in x.strip('[]').split(',')] for x in aug_valid_data.iloc[:,1].values]

# print(train_trajectory_data[0])
# [51841, 52985, 3980, 56289, 56283, 3169, 2642, 29044, 3583, 29045, 29046, 8415, 29046, 8415, 8414]

train_cnt = len(train_trajectory_data)
valid_cnt = len(valid_trajectory_data)

train_td = []
train_atd = []
valid_td = []
valid_atd = []
# n = 0
# m = 0
# max_len1 = 0
# max_len2 = 0

for i in range(train_cnt):
    if (-999 not in train_trajectory_data[i]) and (-999 not in aug_train_trajectory_data[i]):
        if(len(train_trajectory_data[i]) > 20) and (len(train_trajectory_data[i]) < 128):
            # n += 1
            train_td.append(train_trajectory_data[i])
            train_atd.append(aug_train_trajectory_data[i])
            # if len(aug_train_trajectory_data[i]) > max_len1:
            #     max_len1 = len(aug_train_trajectory_data[i])

for i in range(valid_cnt):
    if (-999 not in valid_trajectory_data[i]) and (-999 not in aug_valid_trajectory_data[i]):
        if(len(valid_trajectory_data[i]) > 20) and (len(valid_trajectory_data[i]) < 128):
            # m += 1
            valid_td.append(valid_trajectory_data[i])
            valid_atd.append(aug_valid_trajectory_data[i])
            # if len(aug_valid_trajectory_data[i]) > max_len2:
            #     max_len2 = len(aug_valid_trajectory_data[i])

# print(n)
# print(m)
# # # 200251
# print(len(train_td))
# print(len(train_atd))
# print(len(valid_td))
# print(len(valid_atd))
# print(max_len1)
# print(max_len2)
# 113875
# 16293
# 113875
# 113875
# 16293
# 16293
# 495
# 434
        

def output_trajs(trajs, out_file_path):
    out_file_path.write('id;path\n')
    id = 0
    for traj in trajs:
        out_file_path.write(';'.join([
            str(id), str(traj)
            ]))
        out_file_path.write('\n')
        id += 1

output_trajs(train_td, open("/nas/user/wyh/TNC/data/align_data/norm_SH/SH_train.csv", 'w+'))
output_trajs(train_atd, open("/nas/user/wyh/TNC/data/align_data/norm_SH/SH_aug_train.csv", 'w+'))
output_trajs(valid_td, open("/nas/user/wyh/TNC/data/align_data/norm_SH/SH_valid.csv", 'w+'))
output_trajs(valid_atd, open("/nas/user/wyh/TNC/data/align_data/norm_SH/SH_aug_valid.csv", 'w+'))