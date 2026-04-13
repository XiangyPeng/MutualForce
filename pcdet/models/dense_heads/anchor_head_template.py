import numpy as np
import torch
import torch.nn as nn

from ...utils import box_coder_utils, common_utils, loss_utils
from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.atss_target_assigner import ATSSTargetAssigner
from .target_assigner.axis_aligned_target_assigner import AxisAlignedTargetAssigner
import torch.nn.functional as F

class AnchorHeadTemplate(nn.Module):
    def __init__(self, model_cfg, num_class, class_names, grid_size, point_cloud_range, predict_boxes_when_training):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.use_multihead = self.model_cfg.get('USE_MULTIHEAD', False)

        anchor_target_cfg = self.model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = getattr(box_coder_utils, anchor_target_cfg.BOX_CODER)(
            num_dir_bins=anchor_target_cfg.get('NUM_DIR_BINS', 6),
            **anchor_target_cfg.get('BOX_CODER_CONFIG', {})
        )

        anchor_generator_cfg = self.model_cfg.ANCHOR_GENERATOR_CONFIG
        anchors, self.num_anchors_per_location = self.generate_anchors(
            anchor_generator_cfg, grid_size=grid_size, point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size
        )
        self.anchors = [x.cuda() for x in anchors]
        self.target_assigner = self.get_target_assigner(anchor_target_cfg)

        self.forward_ret_dict = {}
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.criterion = nn.CrossEntropyLoss()

    @staticmethod
    def generate_anchors(anchor_generator_cfg, grid_size, point_cloud_range, anchor_ndim=7):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg
        )
        feature_map_size = [grid_size[:2] // config['feature_map_stride'] for config in anchor_generator_cfg]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros = anchors.new_zeros([*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = torch.cat((anchors, pad_zeros), dim=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def get_target_assigner(self, anchor_target_cfg):
        if anchor_target_cfg.NAME == 'ATSS':
            target_assigner = ATSSTargetAssigner(
                topk=anchor_target_cfg.TOPK,
                box_coder=self.box_coder,
                use_multihead=self.use_multihead,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        elif anchor_target_cfg.NAME == 'AxisAlignedTargetAssigner':
            target_assigner = AxisAlignedTargetAssigner(
                model_cfg=self.model_cfg,
                class_names=self.class_names,
                box_coder=self.box_coder,
                match_height=anchor_target_cfg.MATCH_HEIGHT
            )
        else:
            raise NotImplementedError
        return target_assigner

    def build_losses(self, losses_cfg):
        self.add_module(
            'cls_loss_func',
            loss_utils.SigmoidFocalClassificationLoss(alpha=0.25, gamma=2.0)
        )
        reg_loss_name = 'WeightedSmoothL1Loss' if losses_cfg.get('REG_LOSS_TYPE', None) is None \
            else losses_cfg.REG_LOSS_TYPE
        self.add_module(
            'reg_loss_func',
            getattr(loss_utils, reg_loss_name)(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )
        self.add_module(
            'dir_loss_func',
            loss_utils.WeightedCrossEntropyLoss()
        )

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(
            self.anchors, gt_boxes
        )
        return targets_dict

    def get_cls_layer_loss(self):
        # Network Category Forecast
        cls_preds = self.forward_ret_dict['cls_preds']
        # categories of foreground anchor
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        # Anchor of concern: Get the anchor of foreground and background,
        # and set -1 for anchors with IoU values between 0.45 and 0.6,
        # which are not involved in the loss calculation.
        cared = box_cls_labels >= 0  # [N, num_anchors]
        # The anchor of the foreground
        positives = box_cls_labels > 0
        # The anchor of the background
        negatives = box_cls_labels == 0
        # Assign weights to the anchor of the background.
        negative_cls_weights = negatives * 1.0
        # set the loss weight of each anchor's classification to 1
        cls_weights = (negative_cls_weights + 1.0 * positives).float()
        # set the loss weight of each positive sample anchor regression to 1
        reg_weights = positives.float()
        # When the classification result has only one category
        if self.num_class == 1:
            # class agnostic
            box_cls_labels[positives] = 1

        # The regularization operation is performed and the weight value is calculated,
        # and then the number of positive examples for each data center is calculated.
        pos_normalizer = positives.sum(1, keepdim=True).float()
        # regularize the regression loss with the number of positive samples
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)
        # regularize the classification loss with the number of positive samples
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        # cared contains both foreground and background anchors,
        # and we only need to focus on the category of the foreground part of it,
        # and we do not care whether it is -1 or 0.
        # cared.type_as(box_cls_labels): for the False part of cared, the anchors
        # that are not involved in the loss calculation should all be set to 0.
        # After multiplying the parameters in the corresponding positions,
        # all the anchors between match_threshold and unmatch_threshold are set to 0 for all backgrounds and IoUs.
        cls_targets = box_cls_labels * cared.type_as(box_cls_labels)
        # Perform an extended operation on the last dimension
        cls_targets = cls_targets.unsqueeze(dim=-1)

        cls_targets = cls_targets.squeeze(dim=-1)
        one_hot_targets = torch.zeros(
            *list(cls_targets.shape), self.num_class + 1, dtype=cls_preds.dtype, device=cls_targets.device
        )   # self.num_class + 1: take into account the context.
        
        # A typical application of the scatter_ function is the classification problem,
        # where the labels of the targets are converted to the one-hot encoded form.
        # This part is mainly used in the last dimension,
        # and then the positions of the cls_targets.unsqueeze(dim=-1) indexes are all set to 1.
        one_hot_targets.scatter_(-1, cls_targets.unsqueeze(dim=-1).long(), 1.0) 
        cls_preds = cls_preds.view(batch_size, -1, self.num_class)
        # This procedure does not involve the calculation of background classification loss.
        one_hot_targets = one_hot_targets[..., 1:]
        
        # calculate classification losses
        cls_loss_src = self.cls_loss_func(cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        # sum up and divide by the number of batches
        cls_loss = cls_loss_src.sum() / batch_size
        # loss multiplied by the classification weight
        cls_loss = cls_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['cls_weight']
        tb_dict = {
            'rpn_loss_cls': cls_loss.item()
        }
        return cls_loss, tb_dict

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * torch.sin(boxes2[..., dim:dim + 1])
        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2

    @staticmethod
    def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0, num_bins=2):
        batch_size = reg_targets.shape[0]
        anchors = anchors.view(batch_size, -1, anchors.shape[-1])
        # Considering that reg_targets[... , 6] is the rotation angle obtained after encoding,
        # if you need to convert to the original angle, you need to add the anchor angle back in.
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = common_utils.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        # num_bins = 2
        dir_cls_targets = torch.floor(offset_rot / (2 * np.pi / num_bins)).long()
        dir_cls_targets = torch.clamp(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = torch.zeros(*list(dir_cls_targets.shape), num_bins, dtype=anchors.dtype,
                                      device=dir_cls_targets.device)
            # For one-hot encoding, there are only two directions, forward and reverse.
            dir_targets.scatter_(-1, dir_cls_targets.unsqueeze(dim=-1).long(), 1.0)
            dir_cls_targets = dir_targets
        return dir_cls_targets

    def get_box_reg_layer_loss(self):
        # 7 arguments of anchor_box
        box_preds = self.forward_ret_dict['box_preds']
        # orientation prediction of anchor_box
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        # the coding results for each anchor and GT
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        # get the mask of the foreground anchor in all anchors
        positives = box_cls_labels > 0
        # set the regression parameter to 1
        reg_weights = positives.float()                             # keep only the values with labels greater than 0
        pos_normalizer = positives.sum(1, keepdim=True).float()     # calculate the sum of all positive samples
        reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = torch.cat(
                    [anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1]) for anchor in
                     self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors
        anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        box_preds = box_preds.view(batch_size, -1,
                                   box_preds.shape[-1] // self.num_anchors_per_location if not self.use_multihead else
                                   box_preds.shape[-1])
        # sin(a - b) = sinacosb-cosasinb
        box_preds_sin, reg_targets_sin = self.add_sin_difference(box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, weights=reg_weights)  # [N, M]
        loc_loss = loc_loss_src.sum() / batch_size
        loc_loss = loc_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['loc_weight']
        
        box_loss = loc_loss
        tb_dict = {
            # Use the item() method in PyTorch to return the values of the elements in the tensor.
            'rpn_loss_loc': loc_loss.item()
        }

        # If a directional forecast exists, a directional loss should be added.
        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors, box_reg_targets,
                dir_offset=self.model_cfg.DIR_OFFSET,       # Directional Offset
                num_bins=self.model_cfg.NUM_DIR_BINS        # the number of the direction of BINS
            )
            # the predicted value of the direction
            dir_logits = box_dir_cls_preds.view(batch_size, -1, self.model_cfg.NUM_DIR_BINS)
            # only get the directional prediction of positive samples
            weights = positives.type_as(dir_logits)
            # To make the loss of each sample independent of the number of targets in the sample,
            # the parameter is divided by the number of positive samples.
            weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
            # the calculation of the loss of direction
            dir_loss = self.dir_loss_func(dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            # the calculation of the weight of the loss
            dir_loss = dir_loss * self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS['dir_weight']
            
            # add the loss of direction to the loss of box
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()
        return box_loss, tb_dict



    
    def get_loss_BSH(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss
        
        orignal_heatmap = self.forward_ret_dict['orignal_heatmap']
        feat = self.forward_ret_dict['feat']
        MC_loss = self.MCloss(feat, orignal_heatmap)
        if MC_loss is not None:
            x_contrast_loss = 0.5 * self.MCloss(feat, orignal_heatmap)
   
            # print("error")
            rpn_loss += x_contrast_loss
            tb_dict['x_contrast_loss'] = x_contrast_loss.item()        
        
        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict
    
    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    # 这个是为了 CoIn: Contrastive Instance Feature Mining for Outdoor 3D Object Detectionwith Very Limited Annotations
    # 上面得是原始版本
    # def get_loss(self):
    #     cls_loss, tb_dict = self.get_cls_layer_loss()
    #     box_loss, tb_dict_box = self.get_box_reg_layer_loss()
    #     tb_dict.update(tb_dict_box)
    #     rpn_loss = cls_loss + box_loss
    #     orignal_heatmap = self.forward_ret_dict['orignal_heatmap']
    #     # 这里额外再加一个MC_loss
    #     # EPNet++: Cascade Bi-directional Fusion for Multi-Modal 3D Object Detection
    #     # BEV_loss = self.BEV_loss(self.forward_ret_dict['lidar_batch_spatial_features_bev'],self.forward_ret_dict['radar_batch_spatial_features_bev'])
    #     # BEVContrast: Self-Supervision in BEV Space for Automotive Lidar Point Clouds
    #     # BEV_loss = self.BEV_loss_epnet(self.forward_ret_dict['lidar_batch_spatial_features_bev'],self.forward_ret_dict['radar_batch_spatial_features_bev'])
        
    #     # rpn_loss += 0.5*BEV_loss
        
    #     '''lidar_batch_spatial_features_bev = self.forward_ret_dict['lidar_batch_spatial_features_bev']
    #     lidar_batch_spatial_features_bev_MC_loss = self.MCloss(lidar_batch_spatial_features_bev , orignal_heatmap)
    #     if lidar_batch_spatial_features_bev_MC_loss is not None:
    #         # lidarx_contrast_loss = 0.5 * self.MCloss(feat, orignal_heatmap)
   
    #         # print("error")
    #         rpn_loss += lidar_batch_spatial_features_bev_MC_loss
    #         tb_dict['contrast_BEV_loss_lidar'] = lidar_batch_spatial_features_bev_MC_loss.item()
        
    #     radar_batch_spatial_features_bev = self.forward_ret_dict['radar_batch_spatial_features_bev']
    #     radar_batch_spatial_features_bev_MC_loss = self.MCloss(radar_batch_spatial_features_bev , orignal_heatmap)
    #     if radar_batch_spatial_features_bev_MC_loss is not None:
    #         # lidarx_contrast_loss = 0.5 * self.MCloss(feat, orignal_heatmap)
    #         rpn_loss += radar_batch_spatial_features_bev_MC_loss
    #         tb_dict['contrast_BEV_loss_radar'] = radar_batch_spatial_features_bev_MC_loss.item()
    #     '''
        
    #     # rpn_loss += BEV_loss
    #     # tb_dict['contrast_BEV_loss'] = BEV_loss.item()
        
        
    #     feat = self.forward_ret_dict['feat']
    #     MC_loss = self.MCloss(feat, orignal_heatmap)
    #     if MC_loss is not None:
    #         x_contrast_loss = 0.5 * self.MCloss(feat, orignal_heatmap)
   
    #         # print("error")
    #         rpn_loss += x_contrast_loss
    #         tb_dict['x_contrast_loss'] = x_contrast_loss.item()
   

    #     tb_dict['rpn_loss'] = rpn_loss.item()
    #     return rpn_loss, tb_dict

    def BEV_loss_epnet(self,lidar_bev,radar_bev):
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        # Anchor of concern: Get the anchor of foreground and background,
        # and set -1 for anchors with IoU values between 0.45 and 0.6,
        # which are not involved in the loss calculation.
        cared = box_cls_labels >= 0  # [N, num_anchors]
        # The anchor of the foreground
        positives = box_cls_labels > 0
        # The anchor of the background
        negatives = box_cls_labels == 0
        batch_size = lidar_bev.shape[0] 
        
        P1_1 = F.normalize(lidar_bev.view(batch_size,-1 ,1), p=2, dim=1)
        P2_1 = F.normalize(radar_bev.view(batch_size,-1 ,1), p=2, dim=1)
        
        
        # all_loss= 0.0
        # for i in range(batch_size):
        #     lidar_bev1 = lidar_bev[i,...].permute(1,2,0).view(-1 ,1).contiguous()
        #     radar_bev1 = radar_bev[i,...].permute(1,2,0).view(-1 ,1).contiguous()
        #     aa  =  F.kl_div(lidar_bev1,radar_bev1.detach(), reduction='none')
        #     logits = torch.mm(lidar_bev1, radar_bev1.transpose(1, 0))
        #     target = torch.arange(lidar_bev1.shape[0], device=lidar_bev1.device).long()
        #     out = torch.div(logits, self.temperature)
        #     out = out.contiguous()

        #     loss = self.criterion(out, target)
        #     all_loss += loss
        
        # return all_loss

        batch_size = lidar_bev.shape[0] 
        pos = (box_cls_labels > 0).float().view(-1)# .view(batch_size, -1)
        neg = (box_cls_labels == 0).float().view(-1)# .view(batch_size, -1)
        # F.normalize(lidar_bev.view(batch_size,-1 ,1), p=2, dim=1)
        P1_1 = F.normalize(lidar_bev.view(batch_size,-1 ,1), p=2, dim=-1)  # img prop
        P2_1 = F.normalize(radar_bev.view(batch_size,-1 ,1), p=2, dim=-1)  # point prop
             
        P1 = F.normalize(lidar_bev.view(batch_size,-1 ,1), p=2, dim=1)
        P2 = F.normalize(radar_bev.view(batch_size, -1), dim=-1)
        
        P1 = F.log_softmax(lidar_bev.view(batch_size, -1), dim=-1)  # img prop
        P2 = F.log_softmax(radar_bev.view(batch_size, -1), dim=-1)  # point prop

        P1_1 = F.softmax(lidar_bev.view(batch_size, -1), dim=-1)  # img prop
        P2_1 = F.softmax(radar_bev.view(batch_size, -1), dim=-1)  # point prop
        P = (P1_1.clone() + P2_1.clone()) / 2.0 # B,all
        # 这两个没用
        # kl_loss_p2i = F.kl_div(P1_1, P2_1, reduction='none')
        # mc_loss = 0.5*kl_loss_p2i
        # 下面三行求平均，但是这个我不求平均呢
        kl_loss_i2p = F.kl_div(P1, P.detach(), reduction='none')
        kl_loss_p2i = F.kl_div(P2, P.detach(), reduction='none')
        # mc_loss =  0.5*kl_loss_i2p +  0.5*kl_loss_p2i
        
        mc_loss =  kl_loss_i2p +  kl_loss_p2i
        # 暂时不需要这个
        p1_score = torch.sigmoid(lidar_bev.view(batch_size, -1))
        p2_score = torch.sigmoid(radar_bev.view(batch_size, -1))
        kl_element_weight = (torch.max(p1_score, p2_score) >= 0.2).float()
        # if cfg.ADD_MC_MASK:
        #     kl_element_weight = (torch.max(p1_score, p2_score) >= cfg.MC_MASK_THRES).float()
        # else:
        #     kl_element_weight = torch.ones_like(p1_score)
        # mc_loss = (kl_element_weight.contiguous().view(-1) * mc_loss.contiguous().view(-1) * (pos + neg)).sum()
        mc_loss = ( kl_element_weight.contiguous().view(-1)*mc_loss.contiguous().view(-1)).sum()
        # print(mc_loss)
        return mc_loss 

    def BEV_loss(self,lidar_bev,radar_bev):
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        # Anchor of concern: Get the anchor of foreground and background,
        # and set -1 for anchors with IoU values between 0.45 and 0.6,
        # which are not involved in the loss calculation.
        cared = box_cls_labels >= 0  # [N, num_anchors]
        # The anchor of the foreground
        positives = box_cls_labels > 0
        # The anchor of the background
        negatives = box_cls_labels == 0
        
        
        batch_size = lidar_bev.shape[0] 
        pos = (box_cls_labels > 0).float().view(-1)# .view(batch_size, -1)
        neg = (box_cls_labels == 0).float().view(-1)# .view(batch_size, -1)
        P1 = F.log_softmax(lidar_bev.view(batch_size, -1), dim=-1)  # img prop
        P2 = F.log_softmax(radar_bev.view(batch_size, -1), dim=-1)  # point prop

        P1_1 = F.softmax(lidar_bev.view(batch_size, -1), dim=-1)  # img prop
        P2_1 = F.softmax(radar_bev.view(batch_size, -1), dim=-1)  # point prop
        P = (P1_1.clone() + P2_1.clone()) / 2.0 # B,all
        # 这两个没用
        # kl_loss_p2i = F.kl_div(P1_1, P2_1, reduction='none')
        # mc_loss = 0.5*kl_loss_p2i
        # 下面三行求平均，但是这个我不求平均呢
        kl_loss_i2p = F.kl_div(P1, P.detach(), reduction='none')
        kl_loss_p2i = F.kl_div(P2, P.detach(), reduction='none')
        # mc_loss =  0.5*kl_loss_i2p +  0.5*kl_loss_p2i
        
        mc_loss =  kl_loss_i2p +  kl_loss_p2i
        # 暂时不需要这个
        p1_score = torch.sigmoid(lidar_bev.view(batch_size, -1))
        p2_score = torch.sigmoid(radar_bev.view(batch_size, -1))
        kl_element_weight = (torch.max(p1_score, p2_score) >= 0.2).float()
        # if cfg.ADD_MC_MASK:
        #     kl_element_weight = (torch.max(p1_score, p2_score) >= cfg.MC_MASK_THRES).float()
        # else:
        #     kl_element_weight = torch.ones_like(p1_score)
        # mc_loss = (kl_element_weight.contiguous().view(-1) * mc_loss.contiguous().view(-1) * (pos + neg)).sum()
        mc_loss = ( kl_element_weight.contiguous().view(-1)*mc_loss.contiguous().view(-1)).sum()
        # print(mc_loss)
        return mc_loss 

    def MCloss(self, feat, hm):


        feat = F.normalize(feat, dim=1, p=2)

        bs = hm.size(0)
        h = hm.size(2)
        w = hm.size(3)

        hm_car = hm[:,0,:,:].reshape(bs, 1, h, w)
        hm_ped = hm[:,1,:,:].reshape(bs, 1, h, w)
        hm_cyc = hm[:,2,:,:].reshape(bs, 1, h, w)
        
        hm_sum = hm.sum(1).reshape(bs, 1, h, w)

        # 在真实的heatmap上查找属于1的部分
        tmp_feat_withmask = feat * hm_sum


        # one_ind_car = torch.where(hm_car==1.)
        # one_ind_ped = torch.where(hm_ped==1.)
        # one_ind_cyc = torch.where(hm_cyc==1.)
        
        one_ind_car = torch.where(hm_car>=0.2)
        one_ind_ped = torch.where(hm_ped>=0.2)
        one_ind_cyc = torch.where(hm_cyc>=0.2)
        #one_ind_back = torch.where(hm_sum == 0.)
        
        ####(1)acoording hm to find instance feature
        tmp_feat = tmp_feat_withmask.permute(0,2,3,1).contiguous()

        # 发现对应的car特征 二维索引 torch.Size([12, 3])
        feat_car = tmp_feat[one_ind_car[0],one_ind_car[2],one_ind_car[3]]
        feat_ped = tmp_feat[one_ind_ped[0],one_ind_ped[2],one_ind_ped[3]]
        feat_cyc = tmp_feat[one_ind_cyc[0],one_ind_cyc[2],one_ind_cyc[3]]


        
        feat_car_postive = feat_car.repeat(feat_car.shape[0]-1, 1)
        feat_car_negtive = torch.tensor([]).cuda()
        feat_car_negtive_sub = feat_car.clone()
        for i in range(feat_car.shape[0]-1):
            feat_car_negtive_sub = torch.roll(feat_car_negtive_sub, 1, 0)
            feat_car_negtive = torch.cat((feat_car_negtive, feat_car_negtive_sub), 0)

        num_of_car = feat_car_postive.shape[0]

        feat_ped_postive = feat_ped.repeat(feat_ped.shape[0]-1, 1)
        feat_ped_negtive = torch.tensor([]).cuda()
        feat_ped_negtive_sub = feat_ped.clone()
        for i in range(feat_ped.shape[0]-1):
            feat_ped_negtive_sub = torch.roll(feat_ped_negtive_sub, 1, 0)
            feat_ped_negtive = torch.cat((feat_ped_negtive, feat_ped_negtive_sub), 0)

        num_of_ped = feat_ped_postive.shape[0]

        feat_cyc_postive = feat_cyc.repeat(feat_cyc.shape[0]-1, 1)
        feat_cyc_negtive = torch.tensor([]).cuda()
        feat_cyc_negtive_sub = feat_cyc.clone()
        for i in range(feat_cyc.shape[0]-1):
            feat_cyc_negtive_sub = torch.roll(feat_cyc_negtive_sub, 1, 0)
            feat_cyc_negtive = torch.cat((feat_cyc_negtive, feat_cyc_negtive_sub), 0)


        dims_max = feat_car_postive.shape[0] if feat_car_postive.shape[0] > feat_ped_postive.shape[0] else feat_ped_postive.shape[0]
        dims_max = feat_cyc_postive.shape[0] if feat_cyc_postive.shape[0] > dims_max else dims_max

        if len(feat_car_negtive)!=0:
            feat_car_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_car_postive.shape[0]))(feat_car_postive), 0).reshape(1, -1)
            revert_feat_car_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_car_negtive.shape[0]))(feat_car_negtive), 0).reshape(1, -1)

        else:
            feat_car_flatten = None
            revert_feat_car_flatten =None


        if len(feat_ped_negtive)!=0:
            feat_ped_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_ped_postive.shape[0]))(feat_ped_postive), 0).reshape(1, -1)
            revert_feat_ped_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_ped_negtive.shape[0]))(feat_ped_negtive), 0).reshape(1, -1)
        else:
            feat_ped_flatten = None
            revert_feat_ped_flatten =None

        
        if len(feat_cyc_negtive)!=0:
            feat_cyc_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_cyc_postive.shape[0]))(feat_cyc_postive), 0).reshape(1, -1)
            revert_feat_cyc_flatten = torch.flatten(torch.nn.ZeroPad2d(padding=(0,0,0,dims_max - feat_cyc_negtive.shape[0]))(feat_cyc_negtive), 0).reshape(1, -1)
            
            # assert feat_car_flatten.shape[1] == feat_ped_flatten.shape[1] == feat_cyc_flatten.shape[1] == revert_feat_car_flatten.shape[1] == revert_feat_ped_flatten.shape[1]\
            #     == revert_feat_cyc_flatten.shape[1], "Dimesion unmatch!"
        else:
            feat_cyc_flatten  =None
            revert_feat_cyc_flatten = None


        # else:
        #     assert feat_car_flatten.shape[1] == feat_ped_flatten.shape[1] == revert_feat_car_flatten.shape[1] == revert_feat_ped_flatten.shape[1], "Dimesion unmatch!" 
   
        temp  =[feat_car_flatten, feat_ped_flatten, feat_cyc_flatten]
        temp2  =  [revert_feat_car_flatten,revert_feat_ped_flatten,revert_feat_cyc_flatten]
        get_temp1 = []
        get_temp2 = []
        for i in range(len(temp)):
            if temp[i] is not None:
                get_temp1.append(temp[i])
                get_temp2.append(temp2[i])

        if len(get_temp1)==0 :
            return None

   
        q = torch.cat(get_temp1, 0) 
        
        if len(q.shape)==0:
            return None

        k = torch.cat(get_temp2, 0)

        # q = torch.cat((feat_car_flatten, feat_ped_flatten), 0)
        # k = torch.cat((revert_feat_car_flatten, revert_feat_ped_flatten), 0)


        n = q.size(0)
        # if n!=1:
        logits = torch.mm(q, k.transpose(1, 0))


        logits = logits/ 0.07
        labels = torch.arange(n).cuda().long()
        out = logits.squeeze().contiguous()
        criterion = nn.CrossEntropyLoss().cuda()
        if len(out.shape)!=0:

            loss = criterion(out, labels)      
        else:
            loss = None     
       
            

        return loss


    def generate_predicted_boxes(self, batch_size, cls_preds, box_preds, dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        if isinstance(self.anchors, list):
            # 判断是否需要使用多头预测，默认值为False。
            # Determines if multiple predictions need to be used, the default value is False.
            if self.use_multihead:
                anchors = torch.cat([anchor.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchor.shape[-1])
                                     for anchor in self.anchors], dim=0)
            else:
                anchors = torch.cat(self.anchors, dim=-3)
        else:
            anchors = self.anchors

        # 计算anchor的总数量
        # calculate the total number of anchors
        num_anchors = anchors.view(-1, anchors.shape[-1]).shape[0]
        batch_anchors = anchors.view(1, -1, anchors.shape[-1]).repeat(batch_size, 1, 1)
        # 将预测结果展开为一维的数据张量
        # expand the prediction results into a one-dimensional data tensor
        batch_cls_preds = cls_preds.view(batch_size, num_anchors, -1).float() \
            if not isinstance(cls_preds, list) else cls_preds
        batch_box_preds = box_preds.view(batch_size, num_anchors, -1) if not isinstance(box_preds, list) \
            else torch.cat(box_preds, dim=1).view(batch_size, num_anchors, -1)
        # 解码7个用于预测box的参数
        # decode the 7 parameters used to predict the box
        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, batch_anchors)

        # 对每个anchor的方向的预测
        # prediction of the direction of each anchor
        if dir_cls_preds is not None:
            dir_offset = self.model_cfg.DIR_OFFSET                  # 方向偏移量，0.78539     # direction offset, 0.78539
            dir_limit_offset = self.model_cfg.DIR_LIMIT_OFFSET      # 0
            # 将方向的预测结果展开为一维的张量
            # expand the prediction result of the direction into a one-dimensional tensor
            dir_cls_preds = dir_cls_preds.view(batch_size, num_anchors, -1) if not isinstance(dir_cls_preds, list) \
                else torch.cat(dir_cls_preds, dim=1).view(batch_size, num_anchors, -1)
            # 获取所有anchor的方向分类结果，即正向和反向。
            # Get the directional classification results of all anchors, i.e. forward and reverse.
            dir_labels = torch.max(dir_cls_preds, dim=-1)[1]

            period = (2 * np.pi / self.model_cfg.NUM_DIR_BINS)      # pi
            # 将角度规范在0到pi之间。在OpenPCDet中，坐标方向确定为统一的规范的坐标方向，即x向前，y向左，z向上。
            # Normalize the angle between 0 and pi. In OpenPCDet,
            # the coordinate direction is determined as a uniform canonical coordinate direction,
            # i.e. x forward, y left, z up.
            # 参考训练时的原理，将角度沿着x轴的逆时针方向旋转45度，进而得到dir_rot。
            # Referring to the principle during training,
            # the angle is rotated 45 degrees counterclockwise along the x-axis,
            # which in turn gives dir_rot.
            dir_rot = common_utils.limit_period(
                batch_box_preds[..., 6] - dir_offset, dir_limit_offset, period
            )
            batch_box_preds[..., 6] = dir_rot + dir_offset + period * dir_labels.to(batch_box_preds.dtype)

        if isinstance(self.box_coder, box_coder_utils.PreviousResidualDecoder):
            batch_box_preds[..., 6] = common_utils.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )

        return batch_cls_preds, batch_box_preds

    def forward(self, **kwargs):
        raise NotImplementedError
