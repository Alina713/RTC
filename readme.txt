data_loader中的diff_recnt.cpp
输入：traj roadnet
输出：traj点匹配的路段

traj_deal1.py用于挑选选定区域中的轨迹

source activate torch1.9

ssh wyh@10.176.64.27 -p 30022
服务器数据路径：/nas/user/wyh/dataset/traj
服务器运行路径：
cd nas/user/wyh/pre_contrastive/TNC/traj_dealer
python select_valid_traj_100w.py

【60心跳连接服务器不断】
ssh -o ServerAliveInterval=60 wyh@10.176.64.27 -p 30022
source activate torch1.9

export PYBIND_INCLUDE_PATH=/home/wyh/anaconda3/envs/torch1.9/lib/python3.8/site-packages/pybind11/include/

【坑】
1⃣️
        # 保存所有信息，service for valid_map_trajmm(self) &
        self.info = [] 如果为{}不可以append

        for line in edgeFile.readlines():
            item_list = line.strip().split()
            my_tuple = ()
            my_tuple += tuple(item_list)
2⃣️



【Github 90D】GitHub与pycharm
https://blog.csdn.net/ya6543/article/details/113280085?spm=1001.2014.3001.5506

【git命令】
激活ssh -T git@github.com
 git config --global user.email "zagitova0622@163.com"
 git config --global user.name "Alina713"

commit命令
git add .
git commit -m "全部提交"

git remote add origin https://github.com/Alina713/TNC.git
git branch -M main
git push -u origin main

20230724
map.py 240行传值Roadnet 解决point boost/python


export PYTHONPATH=$PYTHONPATH:/home/wyh/anaconda3/envs/torch1.9/include/pybind11
/home/wyh/myapps/include/pybind11

export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:/home/wyh/anaconda3/envs/torch1.9/include/pybind11

-- Installing: /home/wyh/anaconda3/envs/torch1.9/share/cmake/pybind11/pybind11Targets.cmake

echo "# RTC" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Alina713/RTC.git
git push -u origin main