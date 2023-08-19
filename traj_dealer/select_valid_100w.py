from tqdm import tqdm
import os
# 规定的有效轨迹范围
MINLAT = 31.17491
MINLON = 121.439492
MAXLAT = 31.305073
MAXLON = 121.507001

class Traj:
    def __init__(self, PATH):
        self.trajrec = {}
        self.traj_num = 0
        # self.FLAG = 1
        # trajFile = open("/nas/user/wyh/dataset/traj/Shanghai/20150401_cleaned_mm_trajs.txt")
        trajFile = open(PATH)
        self.trajrec[self.traj_num] = []
        for line in trajFile.readlines():
            item_list = line.strip().split()
            if len(item_list) == 4:
                self.trajrec[self.traj_num].append(item_list)
            else:
                self.traj_num = self.traj_num + 1
                self.trajrec[self.traj_num] = []
                # print(self.trajrec[0])
                # breakpoint()

            # print(self.traj_num)

        # print(self.trajrec[0])

    def get_traj(self):
        return self.trajrec

if __name__ == "__main__":
    print("running")
    # 初始化读取轨迹路径
    file_path = "/nas/user/wyh/dataset/traj/Shanghai"

    n = 0
    file_num = 1
    while(file_num < 10):
        print(file_num)

        filename = '2015040'+str(file_num)+'_cleaned_mm_trajs.txt'
        traj_path = os.path.join(file_path, filename)

        traj = Traj(traj_path).get_traj()
        flag = []
        length = len(traj)

        for i in tqdm(range(length-1)):
            flag.append(1)
            for j in range(len(traj[i])):
                tmplat = float(traj[i][j][1])
                tmplon = float(traj[i][j][2])
                # print(tmplat < MINLAT or tmplat > MAXLAT or tmplon < MINLON or tmplon > MAXLAT)
                # breakpoint()
                if tmplat < MINLAT or tmplat > MAXLAT or tmplon < MINLON or tmplon > MAXLON:
                    flag[i] = 0
                    # print(i)
                    # print(flag[i])
                    # breakpoint()

        # print(flag)
        # breakpoint()

        for a in tqdm(range(length-1)):
            if flag[a] == 0:
                continue
            else:
                for b in range(len(traj[a])):
                    with open("/nas/user/wyh/pre_contrastive/valid_traj_ShangHai.txt", 'a') as f:
                        f.write(traj[a][b][0] + ' ' + traj[a][b][1] + ' ' + traj[a][b][2] + ' ' + traj[a][b][3] + '\n')

            with open("/nas/user/wyh/pre_contrastive/valid_traj_ShangHai.txt", 'a') as f:
                n = n + 1
                f.write('-' + str(n) + '\n')

            # print(traj[i][j][1])
            # breakpoint()

        file_num = file_num + 1








