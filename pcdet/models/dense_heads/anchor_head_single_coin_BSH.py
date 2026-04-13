import numpy as np
import torch.nn as nn

from .anchor_head_template import AnchorHeadTemplate
import torch
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
import torch.nn.functional as F
class AnchorHeadSingle_Coin_BSH(AnchorHeadTemplate):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,voxel_size,
                 predict_boxes_when_training=True):
        super().__init__(
            model_cfg=model_cfg,
            num_class=num_class,
            class_names=class_names,
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        # Three prior frames of different scales exist for each point,
        # and two directions (0 and 90 degrees) exist for each prior frame.
        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        # If there is directional loss, then add a directional convolution layer.
        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()
        
        
        # self.shared_conv = nn.Sequential(
        #     nn.Conv2d(
        #         input_channels, self.model_cfg.SHARED_CONV_CHANNEL, 3, stride=1, padding=1,
        #         bias=self.model_cfg.get('USE_BIAS_BEFORE_NORM', False)
        #     ),
        #     nn.BatchNorm2d(self.model_cfg.SHARED_CONV_CHANNEL),
        #     nn.ReLU(),
        # )
        self.class_names_each_head = []
        self.class_id_mapping_each_head = []
        for cur_class_names in self.model_cfg.CLASS_NAMES_EACH_HEAD:
            self.class_names_each_head.append([x for x in cur_class_names if x in class_names])
            cur_class_id_mapping = torch.from_numpy(np.array(
                [self.class_names.index(x) for x in cur_class_names if x in class_names]
            )).cuda()
            self.class_id_mapping_each_head.append(cur_class_id_mapping)
            
        self.bev_fuse_blocks = nn.Sequential(
            nn.Conv2d(387, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(384, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
            nn.ReLU()
        )


    # initialize parameters
    def init_weights(self):
        pi = 0.01
        # initialize the bias of the classification convolution
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        # initialize the weights of the classification convolution
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

    def forward(self, data_dict):
        # Get the information from the dictionary after backbone processing.
        spatial_features_2d = data_dict['spatial_features_2d']
        self.forward_ret_dict['feat'] = data_dict['hm_prob'] # 这里具体需要再思考
        
        hm_prob  = data_dict['hm_prob']
        
        x = torch.cat((spatial_features_2d, hm_prob), dim=1)
        spatial_features_2d = self.bev_fuse_blocks(x)
        
        
        # 这里用上其中的特征来做一些处理
        
        
        
        
        # 这个是为了coin得mcloss
        # x = self.shared_conv(spatial_features_2d)
        # # torch.Size([2, 64, 160, 160])
        # self.forward_ret_dict['feat'] = x
        # # BEV loss 新得loss
        # if self.training:
        #     self.forward_ret_dict['lidar_batch_spatial_features_bev']=  data_dict['lidar_batch_spatial_features_bev']
        #     self.forward_ret_dict['radar_batch_spatial_features_bev']= data_dict['radar_batch_spatial_features_bev']
    
        # For each coordinate point, there exist 6 category predictions for the prior frame.
        cls_preds = self.conv_cls(spatial_features_2d)
        # For each coordinate point there exist 6 parameter predictions for the a priori frame.
        # Each a priori box needs to predict 7 parameters.
        box_preds = self.conv_box(spatial_features_2d)

        # Adjust the dimension,
        # i.e. swap the data information from the category or parameter adjustment dimension to the last dimension.
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

        # Put the predictions of category and prior frame adjustments into the forward propagation dictionary.
        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds

        # perform prediction of direction classification
        if self.conv_dir_cls is not None:
            # The prediction for each prior frame should be in one of two directions.
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            # Adjust the dimension,
            # i.e. swap the data information from the category or parameter adjustment dimension to the last dimension.
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous() # 每个anchor的方向预测-->(4,12,248,216)
            # Put the direction prediction results into the forward propagation dictionary.
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None
        
        if self.training:
            # 这是原始的
            # targets_dict = {
            #     'box_cls_labels': cls_labels, # (4，321408）
            #     'box_reg_targets': bbox_targets, # (4，321408，7）
            #     'reg_weights': reg_weights # (4，321408）
            # }
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes']
            )
            # Add the results of the GT assignment to the forward propagation dictionary.
            self.forward_ret_dict.update(targets_dict)
            
                    
            target_dict = self.assign_targets2(
                data_dict['gt_boxes'], feature_map_size=spatial_features_2d.size()[2:],
                feature_map_stride=data_dict.get('spatial_features_2d_strides', None)
            )
            #orignal heatmap  torch.Size([2, 3, 160, 160]) 
            self.forward_ret_dict['orignal_heatmap'] = target_dict['heatmaps'][0].clone()
            #update hm
            # target_dict_update = self.batch_update_hm_labels(x, target_dict, thresh=0.6, multi_factor=1)
            # self.forward_ret_dict['target_dicts'] = target_dict_update
            

            

        # self.forward_ret_dict['pred_dicts'] = pred_dicts   
            
        # 如果不是训练模式，则直接进行box的预测或对于双阶段网络要生成proposal(此时batch不为1)
        # If it is not a training model, then box prediction results can be generated directly.
        if not self.training or self.predict_boxes_when_training:
            # Decode and generate the final result based on the prediction results.
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict


    def assign_targets2(self, gt_boxes, feature_map_size=None, **kwargs):
        """
        Args:
            gt_boxes: (B, M, 8)
            range_image_polar: (B, 3, H, W)
            feature_map_size: (2) [H, W]
            spatial_cartesian: (B, 4, H, W)
        Returns:

        """
        feature_map_size = feature_map_size[::-1]  # [H, W] ==> [x, y]
        target_assigner_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        # feature_map_size = self.grid_size[:2] // target_assigner_cfg.FEATURE_MAP_STRIDE

        batch_size = gt_boxes.shape[0]
        ret_dict = {
            'heatmaps': [],
            'target_boxes': [],
            'inds': [],
            'masks': [],
            'heatmap_masks': []
        }

        all_names = np.array(['bg', *self.class_names])
        for idx, cur_class_names in enumerate(self.class_names_each_head): # 对每一个类别
            heatmap_list, target_boxes_list, inds_list, masks_list = [], [], [], []
            for bs_idx in range(batch_size):
                cur_gt_boxes = gt_boxes[bs_idx]
                gt_class_names = all_names[cur_gt_boxes[:, -1].cpu().long().numpy()]

                gt_boxes_single_head = []

                for idx, name in enumerate(gt_class_names):
                    if name not in cur_class_names:
                        continue
                    temp_box = cur_gt_boxes[idx]
                    temp_box[-1] = cur_class_names.index(name) + 1
                    gt_boxes_single_head.append(temp_box[None, :])

                if len(gt_boxes_single_head) == 0:
                    gt_boxes_single_head = cur_gt_boxes[:0, :]
                else:
                    gt_boxes_single_head = torch.cat(gt_boxes_single_head, dim=0)

                heatmap, ret_boxes, inds, mask = self.assign_target_of_single_head(
                    num_classes=len(cur_class_names), gt_boxes=gt_boxes_single_head.cpu(),
                    feature_map_size=feature_map_size, feature_map_stride=target_assigner_cfg.FEATURE_MAP_STRIDE,
                    num_max_objs=target_assigner_cfg.NUM_MAX_OBJS,
                    gaussian_overlap=target_assigner_cfg.GAUSSIAN_OVERLAP,
                    min_radius=target_assigner_cfg.MIN_RADIUS,
                )
                heatmap_list.append(heatmap.to(gt_boxes_single_head.device))
                target_boxes_list.append(ret_boxes.to(gt_boxes_single_head.device))
                inds_list.append(inds.to(gt_boxes_single_head.device))
                masks_list.append(mask.to(gt_boxes_single_head.device))

            ret_dict['heatmaps'].append(torch.stack(heatmap_list, dim=0))
            ret_dict['target_boxes'].append(torch.stack(target_boxes_list, dim=0))
            ret_dict['inds'].append(torch.stack(inds_list, dim=0))
            ret_dict['masks'].append(torch.stack(masks_list, dim=0))
        return ret_dict


    def assign_target_of_single_head(
            self, num_classes, gt_boxes, feature_map_size, feature_map_stride, num_max_objs=500,
            gaussian_overlap=0.1, min_radius=2):
        """
        Args:
            gt_boxes: (N, 8)
            feature_map_size: (2), [x, y]

        Returns:

        """
        heatmap = gt_boxes.new_zeros(num_classes, feature_map_size[1], feature_map_size[0])
        ret_boxes = gt_boxes.new_zeros((num_max_objs, gt_boxes.shape[-1] - 1 + 1))
        inds = gt_boxes.new_zeros(num_max_objs).long()
        mask = gt_boxes.new_zeros(num_max_objs).long()

        x, y, z = gt_boxes[:, 0], gt_boxes[:, 1], gt_boxes[:, 2]
        coord_x = (x - self.point_cloud_range[0]) / self.voxel_size[0] / feature_map_stride
        coord_y = (y - self.point_cloud_range[1]) / self.voxel_size[1] / feature_map_stride
        coord_x = torch.clamp(coord_x, min=0, max=feature_map_size[0] - 0.5)  # bugfixed: 1e-6 does not work for center.int()
        coord_y = torch.clamp(coord_y, min=0, max=feature_map_size[1] - 0.5)  #
        center = torch.cat((coord_x[:, None], coord_y[:, None]), dim=-1)
        center_int = center.int()
        center_int_float = center_int.float()

        dx, dy, dz = gt_boxes[:, 3], gt_boxes[:, 4], gt_boxes[:, 5]
        dx = dx / self.voxel_size[0] / feature_map_stride
        dy = dy / self.voxel_size[1] / feature_map_stride

        radius = centernet_utils.gaussian_radius(dx, dy, min_overlap=gaussian_overlap)
        radius = torch.clamp_min(radius.int(), min=min_radius)

        for k in range(min(num_max_objs, gt_boxes.shape[0])):
            if dx[k] <= 0 or dy[k] <= 0:
                continue

            if not (0 <= center_int[k][0] <= feature_map_size[0] and 0 <= center_int[k][1] <= feature_map_size[1]):
                continue

            cur_class_id = (gt_boxes[k, -1] - 1).long()
            centernet_utils.draw_gaussian_to_heatmap(heatmap[cur_class_id], center[k], radius[k].item())

            inds[k] = center_int[k, 1] * feature_map_size[0] + center_int[k, 0]
            mask[k] = 1

            ret_boxes[k, 0:2] = center[k] - center_int_float[k].float()
            ret_boxes[k, 2] = z[k]
            ret_boxes[k, 3:6] = gt_boxes[k, 3:6].log()
            ret_boxes[k, 6] = torch.cos(gt_boxes[k, 6])
            ret_boxes[k, 7] = torch.sin(gt_boxes[k, 6])
            if gt_boxes.shape[1] > 8:
                ret_boxes[k, 8:] = gt_boxes[k, 7:-1]

        return heatmap, ret_boxes, inds, mask   
    def batch_update_hm_labels(self, x, target_dict, thresh=0.6, multi_factor=1):
        #one2onelable
        thresh = 0.9
        heatmap = target_dict['heatmaps']
        hm = heatmap[0]

        if hm.sum() == 0:
            print("sum of hm is zero")
            return batch, 0,  torch.zeros_like(hm).to(hm)

        ##### prepare the feat
        # normalize the feat
        device =x.device
        feat = x.detach()
        bs = feat.size(0)
        chanel = feat.size(1)
     

        # [bs, 64, h, w]
        #2022.10.5
        feat = F.normalize(feat, dim=1, p=2)
        # [bs, 64, h*w] -> [bs, 64, 1, h*w]
        g_feat = feat.flatten(2).unsqueeze(2)
        # [bs, 1, num_class, h*w]
        weight = hm.flatten(2).unsqueeze(1)
        # obtain each gt instance feature
        # [bs, 64, num_class]
        g_feat = (g_feat * weight).sum(dim=-1)

        ##xqm
        g_feat_car, g_feat_ped, g_feat_cyc = torch.split(g_feat, 1, 2)
        weight_car, weight_ped, weight_cyc = torch.split(weight, 1, 2)
        g_feat_car = g_feat_car / (weight_car.sum())
        g_feat_ped = g_feat_ped / (weight_ped.sum())
        g_feat_cyc = g_feat_cyc / (weight_cyc.sum())
        g_feat = torch.cat((g_feat_car, g_feat_ped, g_feat_cyc), 2)

        # re-normalize
        g_feat = F.normalize(g_feat, dim=1, p=2)
        q_feat = feat.flatten(2).permute(0, 2, 1).contiguous()

 

        #feat = F.normalize(feat, dim=1, p=2)
    
        g_feat_kl = feat.flatten(2).unsqueeze(2)
     
        weight_kl = hm.flatten(2).unsqueeze(1)
        # obtain each gt instance feature
        g_feat_kl = (g_feat_kl * weight_kl).sum(dim=-1)

       
        g_feat_car_kl, g_feat_ped_kl, g_feat_cyc_kl = torch.split(g_feat_kl, 1, 2)
        weight_car_kl, weight_ped_kl, weight_cyc_kl = torch.split(weight_kl, 1, 2)
        g_feat_car_kl = g_feat_car_kl / (weight_car_kl.sum())
        g_feat_ped_kl = g_feat_ped_kl / (weight_ped_kl.sum())
        g_feat_cyc_kl = g_feat_cyc_kl / (weight_cyc_kl.sum())
        g_feat_kl = torch.cat((g_feat_car_kl, g_feat_ped_kl, g_feat_cyc_kl), 2)

        q_feat_kl = feat.flatten(2).permute(0, 2, 1).contiguous()

        dist = torch.bmm(q_feat, g_feat)
        #print(dist.shape)

        #compute l2 distance
        q_feat_kl = q_feat_kl.view(bs, -1, chanel, 1).repeat(1,1,1,3)
        g_feat_kl = g_feat_kl.view(bs, 1, chanel, -1).repeat(1,q_feat.size(1),1,1)
        # dist_l2 = q_feat_l2 - g_feat_l2
        # distl2 = 1. - (torch.sqrt(torch.sum(dist_l2 ** 2, axis=2) + 1e-12)*0.01).clamp_(0., 1.)

        #compute kl
        dist_kl = F.kl_div(q_feat_kl, g_feat_kl, reduction='none', log_target=True)  # [m, nsample, d] - kl(pred, gt) to calculate kl = gt * [ log(gt) - log(pred) ]
        distkl = 1 - self.guiyihua(dist_kl.sum(2))  # [m, nsample]
        

        # [bs, num_class, h, w] -> [bs, h, w]
        sum_hm = hm.sum(dim=1)

        mask_query = torch.where(sum_hm > 0, torch.zeros_like(sum_hm), torch.ones_like(sum_hm)).to(feat)
        mask_query = mask_query.flatten(1).unsqueeze(2)
        mask_dist = dist * mask_query #where is instance where is zero, ba yi biao zhu de wu ti kou diao 

        mask_distkl = distkl* mask_query
        

        # [bs, h*w]
        #value, class_ind = mask_dist_final.max(dim=-1)
        value, class_ind = mask_dist.max(dim=-1)
        valuekl, class_indkl = mask_distkl.max(dim=-1)

        # [bs, h*w]
        t_ind = torch.where(value >= thresh)
        t_indkl = torch.where(valuekl >= thresh)
        
        t_class_ind = class_ind[t_ind]
        t_class_indkl = class_indkl[t_indkl]

        ###### assign the dist to pseudo positive instances
        # [bs, num_class, h*w]
        pseudo_hm = torch.zeros_like(hm).flatten(2).to(feat)
        pseudo_hm[(t_ind[0], t_class_ind, t_ind[1])] = mask_dist[(t_ind[0], t_ind[1], t_class_ind)]
        pseudo_hm = pseudo_hm.view(*hm.size())
        
        pseudo_hm_temp = torch.zeros_like(hm).flatten(2).to(feat)
        pseudo_hm_temp[(t_indkl[0], t_class_indkl, t_indkl[1])] = mask_distkl[(t_indkl[0], t_indkl[1], t_class_indkl)]
        pseudo_hm_temp = pseudo_hm_temp.view(*hm.size())

        # scale the pseudo_hm
        pseudo_hm = (multi_factor * pseudo_hm).clamp_(0., 1.)
        pseudo_hm_temp = (multi_factor * pseudo_hm_temp).clamp_(0., 1.)
        pseudo_hm_final = torch.cat((pseudo_hm[:, 0:1, :,:], pseudo_hm_temp[:, 1:3, :,:]),1)
        hm += pseudo_hm_final
        hm.clamp_(0., 1.)

        target_dict['heatmaps'][0] =  hm


        return target_dict
    
    @staticmethod
    def guiyihua(feature):
        #print(feature.shape)
        value_max, ind_max = feature.max(1)
        value_min, ind_min = feature.min(1)
        feature_guiyihua = (feature - value_min.view(-1, 1 ,3).repeat(1, feature.size(1), 1)) / \
        (value_max.view(-1, 1 ,3).repeat(1, feature.size(1), 1) - value_min.view(-1, 1 ,3).repeat(1, feature.size(1), 1))
        
        return feature_guiyihua