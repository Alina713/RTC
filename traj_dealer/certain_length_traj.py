import pandas as pd

train_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_train.csv", sep=';', header=0)
train_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in train_data.iloc[:,1].values]
train_time_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in train_data.iloc[:, 2].values]

valid_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_valid.csv", sep=';', header=0)
valid_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in valid_data.iloc[:,1].values]
valid_time_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in valid_data.iloc[:, 2].values]

test_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/SHmap_test.csv", sep=';', header=0)
test_trajectory_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in test_data.iloc[:,1].values]
test_time_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in test_data.iloc[:, 2].values]

# for route, time in zip(test_trajectory_data, test_time_data):
#     print(type(route))
#     print(time)
#     break  
# shmap_data = pd.read_csv("/nas/user/wyh/TNC/data/ETA/GAT_transformer/SHmap.csv", sep=';', header=0)
# shmap_traj_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in shmap_data.iloc[:,1].values]
# shmap_time_data = [[int(y) for y in x.strip('[]').split(',') if int(y) != -999] for x in shmap_data.iloc[:,2].values]


def output_traj(routes, times, out_file_path):
    out_file_path.write('id;path;tlist\n')

    id = 0
    for route, time in zip(routes, times):
        # 指示器
        # print(id)
        # print(len(route), len(time))
        # 这里一定要是and 不能是&
        if len(route) > 20 and len(route) < 128:
            out_file_path.write(';'.join([
                str(id), str(route), str(time)
                ]))
            out_file_path.write('\n')
            id += 1
            print(id)


output_traj(train_trajectory_data, train_time_data, open("/nas/user/wyh/TNC/data/ETA/GAT_transformer/SHmap_norm_train.csv", 'w+'))
output_traj(valid_trajectory_data, valid_time_data, open("/nas/user/wyh/TNC/data/ETA/GAT_transformer/SHmap_norm_valid.csv", 'w+'))
output_traj(test_trajectory_data, test_time_data, open("/nas/user/wyh/TNC/data/ETA/GAT_transformer/SHmap_norm_test.csv", 'w+'))

# output_traj(shmap_traj_data, shmap_time_data, open("/nas/user/wyh/TNC/data/ETA/GAT_transformer/SHmap.csv", 'w+'))