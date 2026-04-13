import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ....ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ....ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ....utils import common_utils
from ....ops.roiaware_pool3d import roiaware_pool3d_utils
class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points
    
def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()
    
def weighted_sigmoid_focal_loss(pred,
                                target,
                                weight,
                                gamma=2.0,
                                alpha=0.25,
                                avg_factor=None,
                                num_classes=80):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / num_classes + 1e-6
    return sigmoid_focal_loss(
        pred, target, weight, gamma=gamma, alpha=alpha,
        reduction='sum')[None] / avg_factor

def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    return torch.sum(loss * weight)[None] / avg_factor


def sigmoid_focal_loss(pred,
                       target,
                       weight,
                       gamma=2.0,
                       alpha=0.25,
                       reduction='mean'):
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def bilinear_interpolate_torch(im, x, y):
    """
    Args:
        im: (H, W, C) [y, x]
        x: (N)
        y: (N)

    Returns:

    """
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0 = torch.clamp(x0, 0, im.shape[1] - 1)
    x1 = torch.clamp(x1, 0, im.shape[1] - 1)
    y0 = torch.clamp(y0, 0, im.shape[0] - 1)
    y1 = torch.clamp(y1, 0, im.shape[0] - 1)

    Ia = im[y0, x0]
    Ib = im[y1, x0]
    Ic = im[y0, x1]
    Id = im[y1, x1]

    wa = (x1.type_as(x) - x) * (y1.type_as(y) - y)
    wb = (x1.type_as(x) - x) * (y - y0.type_as(y))
    wc = (x - x0.type_as(x)) * (y1.type_as(y) - y)
    wd = (x - x0.type_as(x)) * (y - y0.type_as(y))
    ans = torch.t((torch.t(Ia) * wa)) + torch.t(torch.t(Ib) * wb) + torch.t(torch.t(Ic) * wc) + torch.t(torch.t(Id) * wd)
    return ans


def sample_points_with_roi(rois, points, sample_radius_with_roi, num_max_points_of_part=200000):
    """
    Args:
        rois: (M, 7 + C)
        points: (N, 3)
        sample_radius_with_roi:
        num_max_points_of_part:

    Returns:
        sampled_points: (N_out, 3)
    """
    if points.shape[0] < num_max_points_of_part:
        distance = (points[:, None, :] - rois[None, :, 0:3]).norm(dim=-1)
        min_dis, min_dis_roi_idx = distance.min(dim=-1)
        roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
        point_mask = min_dis < roi_max_dim + sample_radius_with_roi
    else:
        start_idx = 0
        point_mask_list = []
        while start_idx < points.shape[0]:
            distance = (points[start_idx:start_idx + num_max_points_of_part, None, :] - rois[None, :, 0:3]).norm(dim=-1)
            min_dis, min_dis_roi_idx = distance.min(dim=-1)
            roi_max_dim = (rois[min_dis_roi_idx, 3:6] / 2).norm(dim=-1)
            cur_point_mask = min_dis < roi_max_dim + sample_radius_with_roi
            point_mask_list.append(cur_point_mask)
            start_idx += num_max_points_of_part
        point_mask = torch.cat(point_mask_list, dim=0)

    sampled_points = points[:1] if point_mask.sum() == 0 else points[point_mask, :]

    return sampled_points, point_mask


def sector_fps(points, num_sampled_points, num_sectors):
    """
    Args:
        points: (N, 3)
        num_sampled_points: int
        num_sectors: int

    Returns:
        sampled_points: (N_out, 3)
    """
    sector_size = np.pi * 2 / num_sectors
    point_angles = torch.atan2(points[:, 1], points[:, 0]) + np.pi
    sector_idx = (point_angles / sector_size).floor().clamp(min=0, max=num_sectors)
    xyz_points_list = []
    xyz_batch_cnt = []
    num_sampled_points_list = []
    for k in range(num_sectors):
        mask = (sector_idx == k)
        cur_num_points = mask.sum().item()
        if cur_num_points > 0:
            xyz_points_list.append(points[mask])
            xyz_batch_cnt.append(cur_num_points)
            ratio = cur_num_points / points.shape[0]
            num_sampled_points_list.append(
                min(cur_num_points, math.ceil(ratio * num_sampled_points))
            )

    if len(xyz_batch_cnt) == 0:
        xyz_points_list.append(points)
        xyz_batch_cnt.append(len(points))
        num_sampled_points_list.append(num_sampled_points)
        print(f'Warning: empty sector points detected in SectorFPS: points.shape={points.shape}')

    xyz = torch.cat(xyz_points_list, dim=0)
    xyz_batch_cnt = torch.tensor(xyz_batch_cnt, device=points.device).int()
    sampled_points_batch_cnt = torch.tensor(num_sampled_points_list, device=points.device).int()

    sampled_pt_idxs = pointnet2_stack_utils.stack_farthest_point_sample(
        xyz.contiguous(), xyz_batch_cnt, sampled_points_batch_cnt
    ).long()

    sampled_points = xyz[sampled_pt_idxs]

    return sampled_points


class VoxelSetAbstractionPVRCNN(nn.Module):
    def __init__(self, model_cfg, voxel_size, point_cloud_range, num_bev_features=None,
                 num_rawpoint_features=None, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        SA_cfg = self.model_cfg.SA_LAYER

        self.SA_layers = nn.ModuleList()
        self.SA_layer_names = []
        self.downsample_times_map = {}
        c_in = 0
        for src_name in self.model_cfg.FEATURES_SOURCE:
            if src_name in ['bev', 'raw_points']:
                continue
            self.downsample_times_map[src_name] = SA_cfg[src_name].DOWNSAMPLE_FACTOR

            if SA_cfg[src_name].get('INPUT_CHANNELS', None) is None:
                input_channels = SA_cfg[src_name].MLPS[0][0] \
                    if isinstance(SA_cfg[src_name].MLPS[0], list) else SA_cfg[src_name].MLPS[0]
            else:
                input_channels = SA_cfg[src_name]['INPUT_CHANNELS']

            cur_layer, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
                input_channels=input_channels, config=SA_cfg[src_name]
            )
            self.SA_layers.append(cur_layer)
            self.SA_layer_names.append(src_name)

            c_in += cur_num_c_out

        if 'bev' in self.model_cfg.FEATURES_SOURCE:
            c_bev = num_bev_features
            c_in += c_bev

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            self.SA_rawpoints, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module(
            # self.SA_rawpoints, cur_num_c_out = pointnet2_stack_modules.build_local_aggregation_module_pattern(
                input_channels=num_rawpoint_features - 3, config=SA_cfg['raw_points']
            )

            c_in += cur_num_c_out

        self.vsa_point_feature_fusion = nn.Sequential(
            nn.Linear(c_in, self.model_cfg.NUM_OUTPUT_FEATURES, bias=False),
            nn.BatchNorm1d(self.model_cfg.NUM_OUTPUT_FEATURES),
            nn.ReLU(),
        )
        self.num_point_features = self.model_cfg.NUM_OUTPUT_FEATURES
        self.num_point_features_before_fusion = c_in

        self.voxel_generator = VoxelGeneratorWrapper(
            vsize_xyz=self.voxel_size,
            # vsize_xyz=[0.32, 0.32, 5],
            # vsize_xyz=[0.08, 0.08, 5],
            coors_range_xyz=self.point_cloud_range,
            num_point_features=self.num_point_features,
            max_num_points_per_voxel=10, # 需要适配#config.MAX_POINTS_PER_VOXEL,
            max_num_voxels=15000 # config.MAX_NUMBER_OF_VOXELS[self.mode],
        )  
        # 辅助损失函数
        self.point_cls = nn.Linear(self.num_point_features, 1, bias=False)
        # self.point_reg = nn.Linear(self.num_point_features, 3, bias=False)

        # self.voxel_generator=None
        # self.nz , self.nx , self.ny = 320,320,1 # 可以由前面来计算一下
        self.nz , self.nx , self.ny = 496,432,1 # 可以由前面来计算一下
        self.in_num_bev_features=64
        self.forward_ret_dict={}
        
    def interpolate_from_bev_features(self, keypoints, bev_features, batch_size, bev_stride):
        """
        Args:
            keypoints: (N1 + N2 + ..., 4)
            bev_features: (B, C, H, W)
            batch_size:
            bev_stride:

        Returns:
            point_bev_features: (N1 + N2 + ..., C)
        """
        x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]

        x_idxs = x_idxs / bev_stride
        y_idxs = y_idxs / bev_stride

        point_bev_features_list = []
        for k in range(batch_size):
            bs_mask = (keypoints[:, 0] == k)

            cur_x_idxs = x_idxs[bs_mask]
            cur_y_idxs = y_idxs[bs_mask]
            cur_bev_features = bev_features[k].permute(1, 2, 0)  # (H, W, C)
            point_bev_features = bilinear_interpolate_torch(cur_bev_features, cur_x_idxs, cur_y_idxs)
            point_bev_features_list.append(point_bev_features)

        point_bev_features = torch.cat(point_bev_features_list, dim=0)  # (N1 + N2 + ..., C)
        return point_bev_features

    def sectorized_proposal_centric_sampling(self, roi_boxes, points):
        """
        Args:
            roi_boxes: (M, 7 + C)
            points: (N, 3)

        Returns:
            sampled_points: (N_out, 3)
        """

        sampled_points, _ = sample_points_with_roi(
            rois=roi_boxes, points=points,
            sample_radius_with_roi=self.model_cfg.SPC_SAMPLING.SAMPLE_RADIUS_WITH_ROI,
            num_max_points_of_part=self.model_cfg.SPC_SAMPLING.get('NUM_POINTS_OF_EACH_SAMPLE_PART', 200000)
        )
        sampled_points = sector_fps(
            points=sampled_points, num_sampled_points=self.model_cfg.NUM_KEYPOINTS,
            num_sectors=self.model_cfg.SPC_SAMPLING.NUM_SECTORS
        )
        return sampled_points

    def get_sampled_points(self, batch_dict):
        """
        Args:
            batch_dict:

        Returns:
            keypoints: (N1 + N2 + ..., 4), where 4 indicates [bs_idx, x, y, z]
        """
        batch_size = batch_dict['batch_size']
        if self.model_cfg.POINT_SOURCE == 'raw_points':
            src_points = batch_dict['points'][:, 1:4]
            batch_indices = batch_dict['points'][:, 0].long()
        elif self.model_cfg.POINT_SOURCE == 'voxel_centers':
            src_points = common_utils.get_voxel_centers(
                batch_dict['voxel_coords'][:, 1:4],
                downsample_times=1,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            batch_indices = batch_dict['voxel_coords'][:, 0].long()
        else:
            raise NotImplementedError
        keypoints_list = []
        for bs_idx in range(batch_size):
            bs_mask = (batch_indices == bs_idx)
            sampled_points = src_points[bs_mask].unsqueeze(dim=0)  # (1, N, 3)
            if self.model_cfg.SAMPLE_METHOD == 'FPS':
                cur_pt_idxs = pointnet2_stack_utils.farthest_point_sample(
                    sampled_points[:, :, 0:3].contiguous(), self.model_cfg.NUM_KEYPOINTS
                ).long()

                if sampled_points.shape[1] < self.model_cfg.NUM_KEYPOINTS:
                    times = int(self.model_cfg.NUM_KEYPOINTS / sampled_points.shape[1]) + 1
                    non_empty = cur_pt_idxs[0, :sampled_points.shape[1]]
                    cur_pt_idxs[0] = non_empty.repeat(times)[:self.model_cfg.NUM_KEYPOINTS]

                keypoints = sampled_points[0][cur_pt_idxs[0]].unsqueeze(dim=0)

            elif self.model_cfg.SAMPLE_METHOD == 'SPC':
                cur_keypoints = self.sectorized_proposal_centric_sampling(
                    roi_boxes=batch_dict['rois'][bs_idx], points=sampled_points[0]
                )
                bs_idxs = cur_keypoints.new_ones(cur_keypoints.shape[0]) * bs_idx
                keypoints = torch.cat((bs_idxs[:, None], cur_keypoints), dim=1)
            else:
                raise NotImplementedError

            keypoints_list.append(keypoints)

        keypoints = torch.cat(keypoints_list, dim=0)  # (B, M, 3) or (N1 + N2 + ..., 4)
        if len(keypoints.shape) == 3:
            batch_idx = torch.arange(batch_size, device=keypoints.device).view(-1, 1).repeat(1, keypoints.shape[1]).view(-1, 1)
            keypoints = torch.cat((batch_idx.float(), keypoints.view(-1, 3)), dim=1)

        return keypoints

    def build_aux_target(self, nxyz, gt_boxes3d, enlarge=1.0):
        
        """
        Args:
            points: (N1 + N2 + N3 + ..., 4) [bs_idx, x, y, z]
            gt_boxes: (B, M, 8)
            extend_gt_boxes: [B, M, 8] # 最后一个是类别
        Returns:
            point_cls_labels: (N1 + N2 + N3 + ...), long type, 0:background, -1:ignored
            point_box_labels: (N1 + N2 + N3 + ..., code_size)

        """
        # center_offsets = list()
        # pts_labels = list()
   
        gt_boxes = gt_boxes3d
        points = nxyz
        
        batch_size = gt_boxes.shape[0]
        # 初始化每个点云的类别，默认全0； shape （batch * 16384）
        bs_idx = points[:, 0]
        point_cls_labels = nxyz.new_zeros(points.shape[0]).long() # B*N
        for k in range(batch_size): 
            # 得到一个mask，用于取出一批数据中属于当前帧的点      
            bs_mask = (bs_idx == k)
            # 取出对应的点shape   (16384, 3)
            points_single = points[bs_mask][:, 1:4]
            # 初始化当前帧中点的类别，默认为0    (16384, )
            point_cls_labels_single = point_cls_labels.new_zeros(bs_mask.sum())
            # 计算哪些点在GTbox中, box_idxs_of_pts :return box_idxs_of_pts: (B, M), default background = -1
            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu( # 值是返回点在哪个框里面应该是
                points_single.unsqueeze(dim=0), gt_boxes[k:k + 1, :, 0:7].contiguous()
            ).long().squeeze(dim=0)
             #  # mask 表明该帧中的哪些点属于前景点，哪些点属于背景点;得到属于前景点的mask
            box_fg_flag = (box_idxs_of_pts >= 0  )    # -1为背景点
            point_cls_labels_single[ box_fg_flag] = 1 # 随便搞得
            point_cls_labels[bs_mask] = point_cls_labels_single
        #  shape (batch * N)
        return  point_cls_labels
            
                   
        # 这个偏移量暂时算一不了，后面可以参考IASSD的计算
        '''for i in range(len(gt_boxes3d)):
            boxes3d = gt_boxes3d[i].cpu()
            idx = torch.nonzero(nxyz[:, 0] == i).view(-1)
            new_xyz = nxyz[idx, 1:].cpu()

            boxes3d[:, 3:6] *= enlarge

            pts_in_flag, center_offset = pts_in_boxes3d(new_xyz, boxes3d) # 这个需要单独计算，点是否在框里面
            pts_label = pts_in_flag.max(0)[0].byte()

            # import mayavi.mlab as mlab
            # from mmdet.datasets.kitti_utils import draw_lidar, draw_gt_boxes3d
            # f = draw_lidar((new_xyz).numpy(), show=False)
            # pts = new_xyz[pts_label].numpy()
            # mlab.points3d(pts[:, 0], pts[:, 1], pts[:, 2], color=(1, 1, 1), scale_factor=0.25, figure=f)
            # f = draw_gt_boxes3d(center_to_corner_box3d(boxes3d.numpy()), f, draw_text=False, show=True)

            pts_labels.append(pts_label)
            center_offsets.append(center_offset)
'''
        # center_offsets = torch.cat(center_offsets).cuda()
        # pts_labels = torch.cat(pts_labels).cuda()

        # return pts_labels, center_offsets
    # SA-SSD: 所以这些属性需要直接从self中拿到
    # 方法：在forward中: 提前将需要的值用 self.forward_ret_dict={}保留下来
    # def aux_loss(self, points, point_cls, point_reg, gt_bboxes):
    def get_loss(self, tb_dict=None):
        tb_dict = {}  if tb_dict is None else tb_dict
        # cls_loss, tb_dict = self.get_cls_layer_loss()
        # box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        # tb_dict.update(tb_dict_box)
        # rpn_loss = cls_loss + box_loss

        # tb_dict['rpn_loss'] = rpn_loss.item()
        # return rpn_loss, tb_dict
        '''
         下面两个都是暂时取名字
        '''
        gt_bboxes =  self.forward_ret_dict['gt_bboxes']
        points = self.forward_ret_dict['points'] 
        point_cls = self.forward_ret_dict['point_cls_pred'] 
        N = len(gt_bboxes)
        # return shape (batch * N)
        # pts_labels, center_targets = self.build_aux_target(points, gt_bboxes)
        pts_labels = self.build_aux_target(points, gt_bboxes)

        
        rpn_cls_target = pts_labels.float()
        pos = (pts_labels > 0).float() # 大于0的为前景点
        neg = (pts_labels == 0).float()

        pos_normalizer = pos.sum()
        pos_normalizer = torch.clamp(pos_normalizer, min=1.0)

        cls_weights = pos + neg
        cls_weights = cls_weights / pos_normalizer

        reg_weights = pos
        reg_weights = reg_weights / pos_normalizer

        aux_loss_cls = weighted_sigmoid_focal_loss(point_cls.view(-1), rpn_cls_target, weight=cls_weights, avg_factor=1.)
        aux_loss_cls /= N

        # aux_loss_reg = weighted_smoothl1(point_reg, center_targets, beta=1 / 9., weight=reg_weights[..., None], avg_factor=1.)
        # aux_loss_reg /= N
        
        tb_dict.update({
                'aux_loss_cls': aux_loss_cls.item(),
            })
        return aux_loss_cls,tb_dict
        
        # return dict(
        #     aux_loss_cls = aux_loss_cls,
        #     # aux_loss_reg = aux_loss_reg,
        # )    
        
    


    @staticmethod
    def aggregate_keypoint_features_from_one_source(
            batch_size, aggregate_func, xyz, xyz_features, xyz_bs_idxs, new_xyz, new_xyz_batch_cnt,
            filter_neighbors_with_roi=False, radius_of_neighbor=None, num_max_points_of_part=200000, rois=None
    ):
        """

        Args:
            aggregate_func:
            xyz: (N, 3)
            xyz_features: (N, C)
            xyz_bs_idxs: (N)
            new_xyz: (M, 3)
            new_xyz_batch_cnt: (batch_size), [N1, N2, ...]

            filter_neighbors_with_roi: True/False
            radius_of_neighbor: float
            num_max_points_of_part: int
            rois: (batch_size, num_rois, 7 + C)
        Returns:

        """
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        if filter_neighbors_with_roi: # False
            point_features = torch.cat((xyz, xyz_features), dim=-1) if xyz_features is not None else xyz
            point_features_list = []
            for bs_idx in range(batch_size):
                bs_mask = (xyz_bs_idxs == bs_idx)
                _, valid_mask = sample_points_with_roi(
                    rois=rois[bs_idx], points=xyz[bs_mask],
                    sample_radius_with_roi=radius_of_neighbor, num_max_points_of_part=num_max_points_of_part,
                )
                point_features_list.append(point_features[bs_mask][valid_mask])
                xyz_batch_cnt[bs_idx] = valid_mask.sum()

            valid_point_features = torch.cat(point_features_list, dim=0)
            xyz = valid_point_features[:, 0:3]
            xyz_features = valid_point_features[:, 3:] if xyz_features is not None else None
        else:
            for bs_idx in range(batch_size):
                xyz_batch_cnt[bs_idx] = (xyz_bs_idxs == bs_idx).sum()

        pooled_points, pooled_features = aggregate_func(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz, # Key Point
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=xyz_features.contiguous(),
        )
        return pooled_features

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator



    def transform_points_to_voxels(self,point, data_dict=None, config=None):
        # if data_dict is None:
        #     grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(self.voxel_size)
        #     self.grid_size = np.round(grid_size).astype(np.int64)
        #     # self.voxel_size = config.VOXEL_SIZE
        #     # just bind the config, we will create the VoxelGeneratorWrapper later,
        #     # to avoid pickling issues in multiprocess spawn
        #     return partial(self.transform_points_to_voxels, config=config)

        batch_size = point[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size): 
                       
            batch_mask = point[:, 0] == batch_idx
            this_coords = point[batch_mask, :]         
            voxel_output = self.voxel_generator.generate(this_coords)
            voxels, coordinates, num_points = voxel_output        
        
        
        # data_dict['voxels'] = voxels
        # data_dict['voxels_xyz'] = points[:, 0:3] / config.VOXEL_SIZE
        # data_dict['voxel_coords'] = coordinates
        # data_dict['voxel_num_points'] = num_points   
        
        # voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        voxel_features, voxel_num_points, coords = voxels, num_points , coordinates
        # 每个pillar中最大点云数量
        voxel_count = features.shape[1] 
        # 指明哪些pillar中是需要保留的数据
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)
        # 将feature中被填充数据的所有特征置为0 
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)
        features *= mask
        
        # 这个features其实还可以特征卷积一下，但是这个直接maxpooling   从三维变成2维度   B,nsample,C 变成
        features = torch.max(features, dim=1, keepdim=True)[0]  # 选择BEV里面最大的值作为BEV中的点位置  M,1,3
        features = features.squeeze()  # 这个地方不需要额外特征提取一下吗
        
        # pillar_features, coords = batch_dict['xy_feature'], batch_dict['voxel_coords'] # pillar_features：torch.Size([3563, 64]) batch_size=6
        # spatial features
        batch_spatial_features = []
        indice_xy = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.in_num_bev_features, self.nz * self.nx * self.ny,
                dtype=features.dtype,
                device=features.device)
            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] + this_coords[:, 2] * self.nx + this_coords[:, 3]  # 计算位置
            indices = indices.type(torch.long)
            pillars = features[batch_mask, :] # 拿到
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)
            indice_xy.append(indices)

        batch_spatial_features = torch.stack(batch_spatial_features, 0) # B, C,M
        # reshape回原来空间(伪图像)->(B,C*Z,H,W)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.in_num_bev_features * self.nz, self.ny, self.nx)
        
        return batch_spatial_features
        


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                keypoints: (B, num_keypoints, 3)
                multi_scale_3d_features: {
                        'x_conv4': ...
                    }
                points: optional (N, 1 + 3 + C) [bs_idx, x, y, z, ...]
                spatial_features: optional
                spatial_features_stride: optional

        Returns:
            point_features: (N, C)
            point_coords: (N, 4)

        """
        keypoints = self.get_sampled_points(batch_dict) # keypoints: (N1 + N2 + ..., 4), where 4 indicates [bs_idx, x, y, z]
        # if len(batch_dict['points'])>12000:
        #     print("大于12000点")
        point_features_list = []
        if 'bev' in self.model_cfg.FEATURES_SOURCE:  # 这个可以有，但是需要算一下  
            point_bev_features = self.interpolate_from_bev_features(  #得到串联的BEV+CV特征=spatial_features
                keypoints, batch_dict['spatial_features'], batch_dict['batch_size'],
                # bev_stride=batch_dict['spatial_features_stride'] 
                bev_stride=1
            )
            point_features_list.append(point_bev_features)

        batch_size = batch_dict['batch_size']

        new_xyz = keypoints[:, 1:4].contiguous()
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for k in range(batch_size):
            new_xyz_batch_cnt[k] = (keypoints[:, 0] == k).sum()

        if 'raw_points' in self.model_cfg.FEATURES_SOURCE:
            raw_points = batch_dict['points']
            # 这个地方应该不仅仅是坐标，而是位置和速度都有，可以先位置编码加上速度编码
            pooled_features = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size, aggregate_func=self.SA_rawpoints,
                xyz=raw_points[:, 1:4],
                xyz_features=raw_points[:, 4:].contiguous() if raw_points.shape[1] > 4 else None,
                xyz_bs_idxs=raw_points[:, 0],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt,
                filter_neighbors_with_roi=self.model_cfg.SA_LAYER['raw_points'].get('FILTER_NEIGHBOR_WITH_ROI', False),
                radius_of_neighbor=self.model_cfg.SA_LAYER['raw_points'].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None), # False
                rois=batch_dict.get('rois', None) #None
            )
            point_features_list.append(pooled_features)
        # 这个其实就可以拿着encoder和decoder里面多层卷积来做？？？？
        for k, src_name in enumerate(self.SA_layer_names):
            cur_coords = batch_dict['multi_scale_3d_features'][src_name].indices
            cur_features = batch_dict['multi_scale_3d_features'][src_name].features.contiguous()

            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4], downsample_times=self.downsample_times_map[src_name],
                voxel_size=self.voxel_size, point_cloud_range=self.point_cloud_range
            )

            pooled_features = self.aggregate_keypoint_features_from_one_source(
                batch_size=batch_size, aggregate_func=self.SA_layers[k],
                xyz=xyz.contiguous(), xyz_features=cur_features, xyz_bs_idxs=cur_coords[:, 0],
                new_xyz=new_xyz, new_xyz_batch_cnt=new_xyz_batch_cnt,
                filter_neighbors_with_roi=self.model_cfg.SA_LAYER[src_name].get('FILTER_NEIGHBOR_WITH_ROI', False),
                radius_of_neighbor=self.model_cfg.SA_LAYER[src_name].get('RADIUS_OF_NEIGHBOR_WITH_ROI', None),
                rois=batch_dict.get('rois', None)
            )

            point_features_list.append(pooled_features)

        point_features = torch.cat(point_features_list, dim=-1)

        batch_dict['point_features_before_fusion'] = point_features.view(-1, point_features.shape[-1])# torch.Size([9216, 160])
        point_features = self.vsa_point_feature_fusion(point_features.view(-1, point_features.shape[-1])) # 


         
        if self.training:
            point_features_branch = point_features.view(batch_size,-1,point_features.shape[-1]) # B,N,C
            # point_features_branch = point_features_branch.permute(0,2,1).contiguous()
            point_cls = self.point_cls(point_features_branch) # 2分类
            # point_reg = self.point_reg(point_features_branch) # 中心点回归  nn.Linear(64, 3, bias=False)
            self.forward_ret_dict['point_cls_pred'] = point_cls # torch.Size([6, 1536, 1]) 
            self.forward_ret_dict['points']  = keypoints # torch.Size([9216, 4])
            self.forward_ret_dict['gt_bboxes'] = batch_dict['gt_boxes']# ?  # torch.Size([6, 12, 8])
            # self.get_loss()

        batch_dict['point_features'] = point_features  # (BxN, C) torch.Size([9216, 128])
        batch_dict['point_coords'] = keypoints  # (BxN, 4)
        
        # # 下面是点云拍成PEV,这里可能并不需要点云拍成BEV,或者说把这个Point拍成BEV放到BaseBEVBackboneIntensityDoppler前面去
        # keypoints_feature2 = point_features.view(batch_size,-1, point_features.shape[-1]).contiguous()
        # x_idxs = (keypoints[:, 1] - self.point_cloud_range[0]) / self.voxel_size[0]
        # y_idxs = (keypoints[:, 2] - self.point_cloud_range[1]) / self.voxel_size[1]
        
        # maxx= int((self.point_cloud_range[3]- self.point_cloud_range[0]) / self.voxel_size[0])
        # maxy= int((self.point_cloud_range[4]- self.point_cloud_range[1]) / self.voxel_size[1])
        # r=[]
        # RPVNet中的参考 point-to-range
        # for k in range(batch_size):
        #     bev_features = torch.zeros(maxx, maxy, 128,device=keypoints.device)+1e-5 # torch.Size([2, 256, 188, 188])
        #     bev_features_cumsum = torch.zeros(maxx, maxy, 1,device=keypoints.device)+1e-5 # torch.Size([2, 256, 188, 188])
        #     bs_mask = (keypoints[:, 0] == k)   
        #     cur_x_idxs = torch.floor(x_idxs[bs_mask]).long()# torch.floor(x_idxs[bs_mask],keypoints.devic)  # N,
        #     cur_y_idxs = torch.floor(y_idxs[bs_mask]).long()# torch.floor(y_idxs[bs_mask],keypoints.devic)  # N 
        #     # centerpoint
        #     cur_x_idxs = torch.clamp(cur_x_idxs, max=maxx-1) # torch.Size([1536])
        #     cur_y_idxs = torch.clamp(cur_y_idxs, max=maxy-1) # torch.Size([1536])
        #     # second
        #     # cur_x_idxs = torch.clamp(cur_x_idxs, max=199)
        #     # cur_y_idxs = torch.clamp(cur_y_idxs, max=175)
            
        #     bev_features[cur_x_idxs,cur_y_idxs]=bev_features[cur_x_idxs,cur_y_idxs]+keypoints_feature2[k]
        #     bev_features_cumsum[cur_x_idxs,cur_y_idxs]=bev_features_cumsum[cur_x_idxs,cur_y_idxs]+ torch.ones(size=(keypoints_feature2[k].shape[0],1),device=keypoints.device)  
            
        #     bev_features= bev_features/bev_features_cumsum
        #     r.append(bev_features)
        # batch_dict["Point_BEV_feature"] = torch.stack(r,dim=0).to(device=keypoints.device).view(batch_size,-1, maxy, maxx).contiguous()        
        return batch_dict
