# from pcdet.datasets.once.once_dataset import maintest

# from pcdet.datasets.Roadside.Roadside_dataset import maintest

from pcdet.datasets.kitti.kitti_dataset import maintest
# from pcdet.datasets.astyx.astyx_dataset import maintest

import argparse   

# --func create_once_infos --cfg_file tools/cfgs/dataset_configs/once_dataset.yaml

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
parser.add_argument('--func', type=str, default='create_waymo_infos', help='') # 创建什么格式的
parser.add_argument('--runs_on', type=str, default='server', help='')  # 对应的配置文件
parser.add_argument('--split_name', type=str, default='server', help='') 
args = parser.parse_args()



maintest(args=args)



