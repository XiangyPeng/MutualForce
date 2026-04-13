import copy
import pickle

import numpy as np
from skimage import io

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate

from pcdet.datasets.astyx.object3d_astyx import Object3dAstyx, inv_trans


class KittiDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

        self.kitti_infos = []
        self.include_kitti_data(self.mode)
        #self.pc_type = self.dataset_cfg.POINT_CLOUD_TYPE[0]
        if 'radar' in self.dataset_cfg.POINT_CLOUD_TYPE and 'lidar' in self.dataset_cfg.POINT_CLOUD_TYPE :
            self.pc_type = 'fusion'
        else:
            self.pc_type = self.dataset_cfg.POINT_CLOUD_TYPE[0]
    def include_kitti_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading Vod dataset')
        kitti_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                kitti_infos.extend(infos)

        self.kitti_infos.extend(kitti_infos)

        if self.logger is not None:
            self.logger.info('Total samples for Vod dataset: %d' % (len(kitti_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
    def get_lidar(self, idx):
        lidar_file = self.root_split_path / 'lidar_velodyne' /  ('%s.bin' % idx)
        assert lidar_file.exists()
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        # return np.loadtxt(str(lidar_file), dtype=np.float32, skiprows=1, usecols=(0,1,2,3))
        return points
    def get_radar(self, idx):
        number_of_channels = 7  # ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']
        lidar_file = self.root_split_path / 'radar_velodyne' / ('%s.bin' % idx)
        
        points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, number_of_channels)

        # replace the list values with statistical values; for x, y, z and time, use 0 and 1 as means and std to avoid normalization
        means = [0, 0, 0, 0, 0, 0, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
        stds =  [1, 1, 1, 1, 1, 1, 1]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'

        #in practice, you should use either train, or train+val values to calculate mean and stds. Note that x, y, z, and time are not normed, but you can experiment with that.
        # means = [0, 0, 0, mean_RCS (~ -13.0), mean_v_r (~-3.0), mean_vr_comp (~ -0.1), 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
        # stds =  [1, 1, 1, std_RCS (~14.0),  std_v_r (~8.0),    std_v_r_comp (~6.0), 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
        
        # means = [0, 0, 0, -13.0, -3.0, -0.1, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
        # stds =  [1, 1, 1, 14.0,  8.0,  6.0, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
        

        #we then norm the channels
        points = (points - means)/stds

        return points
        
        # 原始的版本
        # lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
        # assert lidar_file.exists()
        # get_data =np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        # tolerance = 1e-6
        # point_cloud = np.unique(np.round(get_data / tolerance) * tolerance, axis=0)
        # return point_cloud
    # def get_lidar(self, idx):
    #     lidar_file = self.root_split_path / 'velodyne' / ('%s.bin' % idx)
    #     assert lidar_file.exists()
    #     return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    # 这个还需要大改
    def get_pointcloud(self, idx, pc_type):
        if pc_type == 'lidar':
            lidar_file = self.root_split_path / 'lidar_velodyne' /  ('%s.bin' % idx)
            assert lidar_file.exists()
            return np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
            # lidar_file = self.root_split_path / 'lidar_vlp16' / ('%s.txt' % idx)
            # assert lidar_file.exists()
            # return np.loadtxt(str(lidar_file), dtype=np.float32, skiprows=1, usecols=(0, 1, 2, 3))
        elif pc_type == 'radar':
            number_of_channels = 7  # ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']
            lidar_file = self.root_split_path / 'radar_velodyne' / ('%s.bin' % idx)
            
            points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, number_of_channels)

            # replace the list values with statistical values; for x, y, z and time, use 0 and 1 as means and std to avoid normalization
            means = [0, 0, 0, 0, 0, 0, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            stds =  [1, 1, 1, 1, 1, 1, 1]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'

            #in practice, you should use either train, or train+val values to calculate mean and stds. Note that x, y, z, and time are not normed, but you can experiment with that.
            # means = [0, 0, 0, mean_RCS (~ -13.0), mean_v_r (~-3.0), mean_vr_comp (~ -0.1), 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            # stds =  [1, 1, 1, std_RCS (~14.0),  std_v_r (~8.0),    std_v_r_comp (~6.0), 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            
            # means = [0, 0, 0, -13.0, -3.0, -0.1, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            # stds =  [1, 1, 1, 14.0,  8.0,  6.0, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            

            #we then norm the channels
            points = (points - means)/stds
            return points
            # radar_file = self.root_split_path / 'radar_6455' / ('%s.txt' % idx)
            # assert radar_file.exists()
            # return np.loadtxt(str(radar_file), dtype=np.float32, skiprows=2, usecols=(0, 1, 2, 4))
        # 8.15 这里按道理在后续还需要坐标转换 radar 到lidar
        elif pc_type == 'fusion':
            # lidar_file = self.root_split_path / 'lidar_vlp16' / ('%s.txt' % idx)
            # assert lidar_file.exists()
            # lidar_points = np.loadtxt(str(lidar_file), dtype=np.float32, skiprows=1, usecols=(0, 1, 2, 3))
            # radar_file = self.root_split_path / 'radar_6455' / ('%s.txt' % idx)
            # assert radar_file.exists()
            # # 速度信息
            # radar_points = np.loadtxt(str(radar_file), dtype=np.float32, skiprows=2, usecols=(0, 1, 2, 4))
            # x = radar_points[:, 0]
            # z = radar_points[:, 2]
            # x = np.sqrt(x*x + z*z)*np.cos(20/96*np.arctan2(z, x))
            # z = np.sqrt(x*x + z*z)*np.sin(20/96*np.arctan2(z, x))
            # radar_points[:, 0] = x
            # radar_points[:, 2] = z
            # return lidar_points, radar_points
            lidar_file = self.root_split_path / 'lidar_velodyne' /  ('%s.bin' % idx)
            assert lidar_file.exists()
            lidar_points  =  np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)            
            
            number_of_channels = 7  # ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']
            lidar_file = self.root_split_path / 'radar_velodyne' / ('%s.bin' % idx)
            
            points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, number_of_channels)

            # replace the list values with statistical values; for x, y, z and time, use 0 and 1 as means and std to avoid normalization
            means = [0, 0, 0, 0, 0, 0, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            stds =  [1, 1, 1, 1, 1, 1, 1]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'

            #in practice, you should use either train, or train+val values to calculate mean and stds. Note that x, y, z, and time are not normed, but you can experiment with that.
            # means = [0, 0, 0, mean_RCS (~ -13.0), mean_v_r (~-3.0), mean_vr_comp (~ -0.1), 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            # stds =  [1, 1, 1, std_RCS (~14.0),  std_v_r (~8.0),    std_v_r_comp (~6.0), 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            
            # means = [0, 0, 0, -13.0, -3.0, -0.1, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            # stds =  [1, 1, 1, 14.0,  8.0,  6.0, 0]  # 'x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time'
            

            #we then norm the channels
            radar_points = (points - means)/stds       
            
            return lidar_points, radar_points     
            
        # else:
        #     pass
    def get_image(self, idx):
        """
        Loads image for a sample
        Args:
            idx: int, Sample index
        Returns:
            image: (H, W, 3), RGB Image
        """
        img_file = self.root_split_path / 'image_2' / ('%s.jpg' % idx)
        assert img_file.exists()
        image = io.imread(img_file)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def get_image_shape(self, idx):
        img_file = self.root_split_path / 'image_2' / ('%s.jpg' % idx)
        assert img_file.exists()
        return np.array(io.imread(img_file).shape[:2], dtype=np.int32)

    def get_label(self, idx):
        label_file = self.root_split_path / 'label_2' / ('%s.txt' % idx)
        assert label_file.exists()
        return object3d_kitti.get_objects_from_label(label_file)

    def get_depth_map(self, idx):
        """
        Loads depth map for a sample
        Args:
            idx: str, Sample index
        Returns:
            depth: (H, W), Depth map
        """
        depth_file = self.root_split_path / 'depth_2' / ('%s.png' % idx)
        assert depth_file.exists()
        depth = io.imread(depth_file)
        depth = depth.astype(np.float32)
        depth /= 256.0
        return depth
    # lidar和radar都有读取对应的参数
    def get_lidar_calib(self, idx):
        calib_file = self.root_split_path / 'lidar_calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)
    
    def get_radar_calib(self, idx):
        calib_file = self.root_split_path / 'radar_calib' / ('%s.txt' % idx)
        assert calib_file.exists()
        return calibration_kitti.Calibration(calib_file)

    def get_road_plane(self, idx):
        plane_file = self.root_split_path / 'planes' / ('%s.txt' % idx)
        if not plane_file.exists():
            return None

        with open(plane_file, 'r') as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @staticmethod
    def get_fov_flag(pts_rect, img_shape, calib):
        """
        Args:
            pts_rect:
            img_shape:
            calib:

        Returns:

        """
        pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
        val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
        val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
        val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
        pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

        return pts_valid_flag
#    @staticmethod
    # 将坐标和转换方式得到最终转换结果
    def homogeneous_transformation(self,points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """
    This function applies the homogenous transform using the dot product.
        :param points: Points to be transformed in a Nx4 numpy array.
        :param transform: 4x4 transformation matrix in a numpy array.
        :return: Transformed points of shape Nx4 in a numpy array.
        """
        if transform.shape != (4, 4):
            raise ValueError(f"{transform.shape} must be 4x4!")
        if points.shape[1] != 4:
            raise ValueError(f"{points.shape[1]} must be Nx4!")
        return transform.dot(points.T).T

   
    def homogeneous_coordinates(self,points: np.ndarray) -> np.ndarray:
        """
    This function returns the given point array in homogenous coordinates.
        :param points: Input ndarray of shape Nx3.
        :return: Output ndarray of shape Nx4.
        """
        if points.shape[1] != 3:
            raise ValueError(f"{points.shape[1]} must be Nx3!")

        return np.hstack((points,
                        np.ones((points.shape[0], 1),
                                dtype=np.float32)))


    # 利用t_radar_camera和t_camera_lidar参数,得到radar转向lidar坐标系下的值
    def t_radar_lidar(self):
        """
Property which returns the homogeneous transform matrix from the lidar frame, to the radar frame.
        :return: Numpy array of the homogeneous transform matrix from the lidar frame, to the radar frame.
        """
        if self.T_radar_lidar is not None:
            # When the data is already loaded.
            return self.T_radar_lidar
        else:
            # Calculate data if it is calculated yet.
            self.T_radar_lidar = np.dot(self.t_radar_camera, self.t_camera_lidar)
            return self.T_radar_lidar
    
    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % (self.split, sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': self.get_image_shape(sample_idx)}
            info['image'] = image_info
            lidar_calib = self.get_lidar_calib(sample_idx)
            
            radar_calib = self.get_radar_calib(sample_idx)

            P2 = np.concatenate([lidar_calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=lidar_calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = lidar_calib.R0
            V2C_4x4 = np.concatenate([lidar_calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            lidar_calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['lidar_calib'] = lidar_calib_info
            
            
            P2 = np.concatenate([radar_calib.P2, np.array([[0., 0., 0., 1.]])], axis=0)
            R0_4x4 = np.zeros([4, 4], dtype=radar_calib.R0.dtype)
            R0_4x4[3, 3] = 1.
            R0_4x4[:3, :3] = radar_calib.R0
            V2C_4x4 = np.concatenate([radar_calib.V2C, np.array([[0., 0., 0., 1.]])], axis=0)
            radar_calib_info = {'P2': P2, 'R0_rect': R0_4x4, 'Tr_velo_to_cam': V2C_4x4}

            info['radar_calib'] = radar_calib_info

            if has_label:
                obj_list = self.get_label(sample_idx)
                annotations = {}
                annotations['name'] = np.array([obj.cls_type for obj in obj_list])
                annotations['truncated'] = np.array([obj.truncation for obj in obj_list])
                annotations['occluded'] = np.array([obj.occlusion for obj in obj_list])
                annotations['alpha'] = np.array([obj.alpha for obj in obj_list])
                annotations['bbox'] = np.concatenate([obj.box2d.reshape(1, 4) for obj in obj_list], axis=0)
                annotations['dimensions'] = np.array([[obj.l, obj.h, obj.w] for obj in obj_list])  # lhw(camera) format
                annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
                annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])
                annotations['score'] = np.array([obj.score for obj in obj_list])
                annotations['difficulty'] = np.array([obj.level for obj in obj_list], np.int32)

                num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)

                loc = annotations['location'][:num_objects]
                dims = annotations['dimensions'][:num_objects]
                rots = annotations['rotation_y'][:num_objects]
                # 通过计算得到在lidar坐标系下的坐标，loc_lidar:（N,3）
                loc_lidar = lidar_calib.rect_to_lidar(loc)
                # height, width, length 这个是不是不一样
                l, h, w = dims[:, 0:1], dims[:, 1:2], dims[:, 2:3]
                loc_lidar[:, 2] += h[:, 0] / 2
                gt_boxes_lidar = np.concatenate([loc_lidar, l, w, h, -(np.pi / 2 + rots[..., np.newaxis])], axis=1)
                annotations['gt_boxes_lidar'] = gt_boxes_lidar # 其实label就只有lidar的标签

                info['annos'] = annotations

                if count_inside_pts:
                    if self.pc_type == 'lidar' or self.pc_type == 'radar':
                        points = self.get_pointcloud(sample_idx, self.pc_type)
                        # 这个是单独的呢还是所有呢？
                        # points = self.get_lidar(sample_idx) # 这个应该去掉把，正常前面那个就可以
                        calib = self.get_lidar_calib(sample_idx)
                        # # 将lidar坐标系的点变换到rect坐标系 有可能lidar或radar
                        pts_rect_points = calib.lidar_to_rect(points[:, 0:3])
                        
                        fov_flag = self.get_fov_flag(pts_rect_points, info['image']['image_shape'], calib)
                        pts_fov = points[fov_flag]
                        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)

                        
                        
                    else:
                        # if sample_idx=="00544":
                        #     print("sdsd")
                        # lidar_points, radar_points = self.get_pointcloud(sample_idx, self.pc_type)
                        # 这个是单独的呢还是所有呢？
                        lidar_points = self.get_lidar(sample_idx) # 这个应该是有问题的吧，应该是lidar和radar里面总共的点数目
                        lidar_calib = self.get_lidar_calib(sample_idx)
                        pts_rect_lidar_points = lidar_calib.lidar_to_rect(lidar_points[:, 0:3])
                        
                        
                        # 返回true or false list判断点云是否在fov下，判断该点云能否有效 （是否用于训练）
                        fov_flag = self.get_fov_flag(pts_rect_lidar_points, info['image']['image_shape'], lidar_calib)
                        # 提取有效点
                        pts_fov = lidar_points[fov_flag]
                        # gt_boxes_lidar是(N,7)  [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
                        # 返回值corners_lidar为（N,8,3）
                        corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                        
                        #  8.15：这里可能还需要再处理
                        radar_points = self.get_radar(sample_idx)
                        radar_calib = self.get_radar_calib(sample_idx)
                        pts_rect_radar_points = radar_calib.lidar_to_rect( radar_points[:, 0:3])
                    
                    
                    
                    if self.pc_type == 'lidar' or self.pc_type == 'radar':
                        num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    else:
                        lidar_num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                        radar_num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                        # 我觉得无所谓，因为只有lidar有标签
                        num_points_in_gt = -np.ones(num_gt, dtype=np.int32)
                    
                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()
                        
                    # fov_flag = self.get_fov_flag(lidar_points, info['image']['image_shape'], calib)
                    # pts_fov = points[fov_flag]
                    ''' 8.15 注释
                    corners = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    
                    if self.pc_type == 'lidar' or self.pc_type == 'radar':                                           
                        for k in range(num_objects):
                            flag = box_utils.in_hull(points[:, 0:3], corners[k])
                            num_points_in_gt[k] = flag.sum()
                    else:
                        for k in range(num_objects):
                            # lidar_flag = box_utils.in_hull(lidar_points[:, 0:3], corners[k])
                            lidar_flag = box_utils.in_hull(lidar_points[:, 0:3], corners[k])
                            
                            lidar_num_points_in_gt[k] = lidar_flag.sum()
                            # radar_flag = box_utils.in_hull(radar_points[:, 0:3], corners[k])
                            radar_flag = box_utils.in_hull(radar_points[:, 0:3], corners[k])
                            radar_num_points_in_gt[k] = radar_flag.sum()
                    if self.pc_type == 'lidar' or self.pc_type == 'radar':
                        annotations['num_points_in_gt'] = num_points_in_gt
                    else:
                        annotations['lidar_num_points_in_gt'] = lidar_num_points_in_gt
                        annotations['radar_num_points_in_gt'] = radar_num_points_in_gt  
                        annotations['num_points_in_gt'] = num_points_in_gt # 8.13添加，视为lidar为主要参考
                    '''
            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
    # 用trainfile的groundtruth产生groundtruth_database，
    # 只保存训练数据中的gt_box及其包围的点的信息，用于数据增强
    def create_groundtruth_database(self, info_path=None, used_classes=None, split='train'):
        import torch
        # import Path
        from pathlib import Path
        database_save_path = Path(self.root_path) / ('gt_database' if split == 'train' else ('gt_database_%s' % split))
        db_info_save_path = Path(self.root_path) / ('kitti_dbinfos_%s.pkl' % split)

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
        # 读取infos里的每个info的信息，一个info是一帧的数据
        for k in range(len(infos)):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]
            sample_idx = info['point_cloud']['lidar_idx']
            if self.pc_type == 'lidar' or self.pc_type == 'radar':
                points = self.get_pointcloud(sample_idx, self.pc_type)
            else:
                # 这里是将lidar的坐标系转了
                lidar_points, radar_points = self.get_pointcloud(sample_idx, self.pc_type)
            
            # points = self.get_lidar(sample_idx)
            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            bbox = annos['bbox']
            gt_boxes = annos['gt_boxes_lidar']
            # num_obj是有效物体的个数，为N
            num_obj = gt_boxes.shape[0]
            
            # 返回每个box中的点云索引[0 0 0 1 0 1 1...]
            if self.pc_type == 'lidar' or self.pc_type == 'radar':
                point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints)
            else:
                lidar_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(lidar_points[:, 0:3]), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints)
                # 8.15 这个应该有问题，坐标没对应
                radar_point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                    torch.from_numpy(radar_points[:, 0:3]), torch.from_numpy(gt_boxes)
                ).numpy()  # (nboxes, npoints)
             
            # point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
            #     torch.from_numpy(points[:, 0:3]), torch.from_numpy(gt_boxes)
            # ).numpy()  # (nboxes, npoints)
            if self.pc_type == 'lidar' or self.pc_type == 'radar':
                for i in range(num_obj):
                    filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                    filepath = database_save_path / filename
                    gt_points = points[point_indices[i] > 0]

                    gt_points[:, :3] -= gt_boxes[i, :3]
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    if (used_classes is None) or names[i] in used_classes:
                        db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        db_info = {'name': names[i], 'path': db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'num_points_in_gt': gt_points.shape[0],
                                'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                        if names[i] in all_db_infos:
                            all_db_infos[names[i]].append(db_info)
                        else:
                            all_db_infos[names[i]] = [db_info]
            else:
                for i in range(num_obj):
                    # filename = '%s_%s_%d.bin' % (sample_idx, names[i], i)
                    lidar_filename = '%s_%s_%s_%d.bin' % ('lidar', sample_idx, names[i], i)
                    radar_filename = '%s_%s_%s_%d.bin' % ('radar', sample_idx, names[i], i)
                    lidar_filepath = database_save_path / lidar_filename
                    radar_filepath = database_save_path / radar_filename
                    # point_indices[i] > 0得到的是一个[T,F,T,T,F...]之类的真假索引，共有M个
                    # 再从points中取出相应为true的点云数据，放在gt_points中 相对于属于gt的点云拿出来
                    lidar_gt_points = lidar_points[lidar_point_indices[i] > 0]
                    radar_gt_points = radar_points[radar_point_indices[i] > 0]
                    # gt_points = np.concatenate((lidar_gt_points, radar_gt_points), axis=0)

                    # gt_points[:, :3] -= gt_boxes[i, :3]
                    lidar_gt_points[:, :3] -= gt_boxes[i, :3]
                    radar_gt_points[:, :3] -= gt_boxes[i, :3]
                    # with open(filepath, 'w') as f:
                    #     gt_points.tofile(f)
                    with open(lidar_filepath, 'w') as f:
                         lidar_gt_points.tofile(f)
                    with open(radar_filepath, 'w') as f:
                         radar_gt_points.tofile(f)

                    if (used_classes is None) or names[i] in used_classes:
                        # db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        lidar_db_path = str(lidar_filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        radar_db_path = str(radar_filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                        db_info = {'name': names[i], 'lidar_path': lidar_db_path, 'radar_path': radar_db_path, 'image_idx': sample_idx, 'gt_idx': i,
                                'box3d_lidar': gt_boxes[i], 'lidar_num_points_in_gt': lidar_gt_points.shape[0], 'radar_num_points_in_gt': radar_gt_points.shape[0],
                                'difficulty': difficulty[i], 'bbox': bbox[i], 'score': annos['score'][i]}
                        if names[i] in all_db_infos:
                            all_db_infos[names[i]].append(db_info)
                        else:
                            all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            calib = batch_dict['calib'][batch_index]
            # image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
            image_shape = batch_dict['image_shape'][batch_index]
            pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .vod.evaluation import evaluate  as vod_eval 
        from .kitti_object_eval_python import eval as kitti_eval
        # vod 数据集的
        import os
        eval_det_annos2 = copy.deepcopy(det_annos)
        eval_gt_annos2 = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        test_annotation_file= os.path.join("/mnt/data/tm/code/Radar/detection/Delft_dataset_pcdet/data/view_of_delft_PUBLIC/radar_5frames/training", 'label')
        vod_eval_Evaluation = vod_eval.Evaluation(test_annotation_file)
        result_path = os.path.join("/mnt/data/tm/code/Radar/detection/Delft_dataset_pcdet/data/view_of_delft_PUBLIC/radar_5frames/training",'detection')
        # results = vod_eval_Evaluation.evaluate(result_path=result_path,current_class=[0, 1, 2])
        results = vod_eval_Evaluation.evaluate(eval_gt_annos2, eval_det_annos2, class_names,current_class=[0, 1, 2])

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names) # 直接就计算了，不需要图像
        
        out_str_pattern = "Results: \n"+"Entire annotated area: \n"+"Car: {} \n".format(str(results['entire_area']['Car_3d_all']))+"Pedestrian: {} \n".format(str(results['entire_area']['Pedestrian_3d_all']))+"Cyclist: {} \n".format(str(results['entire_area']['Cyclist_3d_all']))+"mAP: {} \n".format(str((results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3))+"Driving corridor area: \n"+"Car: {} \n".format(str(results['roi']['Car_3d_all']))+"Pedestrian: {} \n".format(str(results['roi']['Pedestrian_3d_all']))+"Cyclist: {} \n".format(str(results['roi']['Cyclist_3d_all']))+"mAP: {} \n".format(str((results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3))
      
        print("Results: \n"
            f"Entire annotated area: \n"
            f"Car: {results['entire_area']['Car_3d_all']} \n"
            f"Pedestrian: {results['entire_area']['Pedestrian_3d_all']} \n"
            f"Cyclist: {results['entire_area']['Cyclist_3d_all']} \n"
            f"mAP: {(results['entire_area']['Car_3d_all'] + results['entire_area']['Pedestrian_3d_all'] + results['entire_area']['Cyclist_3d_all']) / 3} \n"
            f"Driving corridor area: \n"
            f"Car: {results['roi']['Car_3d_all']} \n"
            f"Pedestrian: {results['roi']['Pedestrian_3d_all']} \n"
            f"Cyclist: {results['roi']['Cyclist_3d_all']} \n"
            f"mAP: {(results['roi']['Car_3d_all'] + results['roi']['Pedestrian_3d_all'] + results['roi']['Cyclist_3d_all']) / 3} \n"
            )
        
        return ap_result_str, ap_dict,out_str_pattern

        # eval_det_annos = copy.deepcopy(det_annos)
        # eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        # ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        # return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.kitti_infos) * self.total_epochs

        return len(self.kitti_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.kitti_infos)

        info = copy.deepcopy(self.kitti_infos[index])
        # 
        sample_idx = info['point_cloud']['lidar_idx']
        img_shape = info['image']['image_shape']
        lidar_calib = self.get_lidar_calib(sample_idx)
        radar_calib = self.get_radar_calib(sample_idx)
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])        
        
        input_dict = {
            'frame_id': sample_idx,
            'calib': lidar_calib, # 8.15 这里我现在默认是lidar版本的，但是对单radar不友好            'lidar_calib': lidar_calib, 
            'radar_calib': radar_calib,
            'lidar_calib': lidar_calib,
            
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            # 8.15 lidar坐标系，所以按道理radr也得到lidar坐标系下
            gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
            gt_boxes_lidar = box_utils.boxes3d_kitti_camera_to_lidar(gt_boxes_camera, lidar_calib)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
            if "gt_boxes2d" in get_item_list:
                input_dict['gt_boxes2d'] = annos["bbox"]

            road_plane = self.get_road_plane(sample_idx)
            if road_plane is not None:
                input_dict['road_plane'] = road_plane
        
        if "points" in get_item_list:
            if self.pc_type == 'lidar' or self.pc_type == 'radar':
                points = self.get_pointcloud(sample_idx, self.pc_type)
                # points = self.get_lidar(sample_idx)
                # 8.15 这个地方有问题，应该radar也有的
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    
                    '''pts_rect = calib.lidar_to_rect(points[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape, calib)
                    points = points[fov_flag]'''
                    # 8.15 稍微改了一下
                    if self.pc_type == 'lidar':
                        pts_rect = lidar_calib.lidar_to_rect(points[:, 0:3])
                        fov_flag = self.get_fov_flag(pts_rect, img_shape, lidar_calib)
                        points = points[fov_flag]
                    elif self.pc_type == 'radar':
                        pts_rect = radar_calib.lidar_to_rect(points[:, 0:3])
                        fov_flag = self.get_fov_flag(pts_rect, img_shape, radar_calib)
                        points = points[fov_flag]
                    
                input_dict['points'] = points
            else:
                lidar_points, radar_points = self.get_pointcloud(sample_idx, self.pc_type)
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = lidar_calib.lidar_to_rect(lidar_points[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape, lidar_calib)
                    lidar_points =lidar_points[fov_flag]
                input_dict['lidar_points'] = lidar_points
                
                if self.dataset_cfg.FOV_POINTS_ONLY:
                    pts_rect = radar_calib.lidar_to_rect(radar_points[:, 0:3])
                    fov_flag = self.get_fov_flag(pts_rect, img_shape, radar_calib)
                    radar_points =radar_points[fov_flag]
                input_dict['radar_points'] = radar_points


        if "images" in get_item_list:
            input_dict['images'] = self.get_image(sample_idx)

        if "depth_maps" in get_item_list:
            input_dict['depth_maps'] = self.get_depth_map(sample_idx)

        if "calib_matricies" in get_item_list:
            input_dict["trans_lidar_to_cam"], input_dict["trans_cam_to_img"] = kitti_utils.calib_to_matricies(calib)
        # 8.15： 和上面对应，后面应该还是得该calib取消掉
        input_dict['calib'] = lidar_calib
        input_dict['lidar_calib'] = lidar_calib
        input_dict['radar_calib'] = radar_calib
        data_dict = self.prepare_data(data_dict=input_dict)

        data_dict['image_shape'] = img_shape
        return data_dict


def create_kitti_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = KittiDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('kitti_infos_%s.pkl' % train_split)
    val_filename = save_path / ('kitti_infos_%s.pkl' % val_split)
    trainval_filename = save_path / 'kitti_infos_trainval.pkl'
    test_filename = save_path / 'kitti_infos_test.pkl'

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers, has_label=True, count_inside_pts=True)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    with open(trainval_filename, 'wb') as f:
        pickle.dump(kitti_infos_train + kitti_infos_val, f)
    print('Kitti info trainval file is saved to %s' % trainval_filename)

    dataset.set_split('test')
    kitti_infos_test = dataset.get_infos(num_workers=workers, has_label=False, count_inside_pts=False)
    with open(test_filename, 'wb') as f:
        pickle.dump(kitti_infos_test, f)
    print('Kitti info test file is saved to %s' % test_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(train_filename, split=train_split)

    print('---------------Data preparation Done---------------')

def maintest(args):
    # import argparse   

    # # --func create_once_infos --cfg_file tools/cfgs/dataset_configs/once_dataset.yaml
    
    # parser = argparse.ArgumentParser(description='arg parser')
    # parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    # parser.add_argument('--func', type=str, default='create_waymo_infos', help='') # 创建什么格式的
    # parser.add_argument('--runs_on', type=str, default='server', help='')  # 对应的配置文件
    # args = parser.parse_args()

    if args.func == 'create_vod_infos':  
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(args.cfg_file)))  #加载配置我呢见


        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        once_data_path = ROOT_DIR / 'data' / 'view_of_delft_PUBLIC'
        once_save_path = ROOT_DIR / 'data' / 'view_of_delft_PUBLIC'
        # 这个没有
        if args.runs_on == 'cloud':
            once_data_path = Path('/cache/view_of_delft_PUBLIC/')
            once_save_path = Path('/cache/view_of_delft_PUBLIC/')
            dataset_cfg.DATA_PATH = dataset_cfg.CLOUD_DATA_PATH

        create_kitti_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Car', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'view_of_delft_PUBLIC'/'radar_5frames',
            save_path=ROOT_DIR / 'data' / 'view_of_delft_PUBLIC'/'radar_5frames'
        )

# if __name__ == '__main__':
#     import sys
#     if sys.argv.__len__() > 1 and sys.argv[1] == 'create_kitti_infos':
#         import yaml
#         from pathlib import Path
#         from easydict import EasyDict
#         dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
#         ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
#         create_kitti_infos(
#             dataset_cfg=dataset_cfg,
#             class_names=['Car', 'Pedestrian', 'Cyclist'],
#             data_path=ROOT_DIR / 'data' / 'kitti',
#             save_path=ROOT_DIR / 'data' / 'kitti'
#         )