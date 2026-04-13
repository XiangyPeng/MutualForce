import torch
import torch.nn as nn
import torch.nn.functional as F

from .vfe_template import VFETemplate

# This section is similar to a simplified version of PointNet.
class interRAL(nn.Module):
    def __init__(self, channels):
        super(interRAL, self).__init__()
        self.linear = nn.Linear(10, channels, bias=True)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y):
        x = self.linear(x).permute(0, 2, 1)
        y = self.linear(y).permute(0, 2, 1)
        x_q = self.q_conv(x).permute(2, 0, 1) # b, n, c 
        y_k = self.k_conv(y).permute(2, 1, 0)# b, c, n        
        y_v = self.v_conv(y).permute(2, 0, 1)
        energy = torch.bmm(x_q, y_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        y_r = torch.bmm(attention, y_v).permute(1, 2, 0) # b, c, n 
        y_r = self.act(self.after_norm(self.trans_conv(x - y_r)))
        x = x + y_r
        x = torch.max(x, dim=2, keepdim=True)[0]
        return x



# This section is similar to a simplified version of PointNet.
class interRAL_Lidar_other(nn.Module):
    def __init__(self, channels):
        super(interRAL_Lidar_other, self).__init__()
        self.linear = nn.Linear(64, channels, bias=True)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y): # x: N1,M,C  y:N2,M,C
        x = self.linear(x).permute(0, 2, 1)
        y = self.linear(y).permute(0, 2, 1)
        x_q = self.q_conv(x).permute(2, 0, 1) # b, n, c 
        y_k = self.k_conv(y).permute(2, 1, 0)# b, c, n        
        y_v = self.v_conv(y).permute(2, 0, 1)
        energy = torch.bmm(x_q, y_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        y_r = torch.bmm(attention, y_v).permute(1, 2, 0) # b, c, n 
        y_r = self.act(self.after_norm(self.trans_conv(x - y_r)))
        x = x + y_r
        # x = torch.max(x, dim=2, keepdim=True)[0]
        return x

# This section is similar to a simplified version of PointNet.
class interRAL_other(nn.Module):
    def __init__(self, channels):
        super(interRAL_other, self).__init__()
        self.linear = nn.Linear(10, channels, bias=True)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y): # x: N1,M,C  y:N2,M,C
        x = self.linear(x).permute(0, 2, 1)
        y = self.linear(y).permute(0, 2, 1)
        x_q = self.q_conv(x).permute(2, 0, 1) # b, n, c 
        y_k = self.k_conv(y).permute(2, 1, 0)# b, c, n        
        y_v = self.v_conv(y).permute(2, 0, 1)
        energy = torch.bmm(x_q, y_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        y_r = torch.bmm(attention, y_v).permute(1, 2, 0) # b, c, n 
        y_r = self.act(self.after_norm(self.trans_conv(x - y_r)))
        x = x + y_r
        # x = torch.max(x, dim=2, keepdim=True)[0]
        return x

# This section is similar to a simplified version of PointNet.
class interRAL_other_lidar_voc(nn.Module):
    def __init__(self, channels):
        super(interRAL_other_lidar_voc, self).__init__()
        self.linear = nn.Linear(64, channels, bias=True)
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight 
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, y): # x: N1,M,C  y:N2,M,C
        x = self.linear(x).permute(0, 2, 1)
        y = self.linear(y).permute(0, 2, 1)
        x_q = self.q_conv(x).permute(2, 0, 1) # b, n, c 
        y_k = self.k_conv(y).permute(2, 1, 0)# b, c, n        
        y_v = self.v_conv(y).permute(2, 0, 1)
        energy = torch.bmm(x_q, y_k) # b, n, n 
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        y_r = torch.bmm(attention, y_v).permute(1, 2, 0) # b, c, n 
        y_r = self.act(self.after_norm(self.trans_conv(x - y_r)))
        x = x + y_r
        # x = torch.max(x, dim=2, keepdim=True)[0]
        return x


class PFNLayer_vel(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)      # ascend dimension with 64 output channels
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)   # 1D BN layer
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)     # ascend the demension
        torch.backends.cudnn.enabled = False
        # change dimensions, (pillars,num_points,channels) -> (pillars,channels,num_points)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        return x
        # maxpool operation, and find the point in each pillar that best represents the pillar
        # x_max = torch.max(x, dim=1, keepdim=True)[0]

        # if self.last_vfe:
        #     # return the results obtained by processing the pillar from a simplified version of PointNet
        #     return x_max
        # else:
        #     x_repeat = x_max.repeat(1, inputs.shape[1], 1)
        #     x_concatenated = torch.cat([x, x_repeat], dim=2)
        #     return x_concatenated



class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)      # ascend dimension with 64 output channels
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)   # 1D BN layer
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            x = self.linear(inputs)     # ascend the demension
        torch.backends.cudnn.enabled = False
        # change dimensions, (pillars,num_points,channels) -> (pillars,channels,num_points)
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        x = F.relu(x)
        # maxpool operation, and find the point in each pillar that best represents the pillar
        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            # return the results obtained by processing the pillar from a simplified version of PointNet
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated



class PillarVFE_Lidar(VFETemplate):
    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range, **kwargs):
        super().__init__(model_cfg=model_cfg)
        num_point_features =4
        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        if 'lidar_voxels' in batch_dict:
            batch_dict['voxels'] = batch_dict['lidar_voxels']
            batch_dict['voxel_num_points']= batch_dict['lidar_voxel_num_points']
            batch_dict['voxel_coords'] = batch_dict['lidar_voxel_coords']
        voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
        # Summing all point clouds in each pillar.
        # if keepdim=True is set, the original dimension information will be kept.
        # Divide the summation information by the number of points in each point cloud to get the average of all point clouds in each pillar.
        points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
        
        # Subtract the average value of the corresponding pillar from each point cloud data to get the difference.
        f_cluster = voxel_features[:, :, :3] - points_mean
        
        # Resume the null data for each point cloud to the centroid offset of this pillar coordinate.
        f_center = torch.zeros_like(voxel_features[:, :, :3])

        # The coordinates (coords) of each grid point multiplied by the length and width of each pillar,
        # then we can obtain the actual length and width of the point cloud data (in m).
        # Add half of the length and width of each pillar to obtain the centroid coordinates of each pillar.
        # Subtract the centroid coordinates of the corresponding pillar from the x, y, and z of each point,
        # then we get the offset from each point to the centroid of the corresponding each point.
        f_center[:, :, 0] = voxel_features[:, :, 0] - (
                        coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
        f_center[:, :, 1] = voxel_features[:, :, 1] - (
                        coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
        f_center[:, :, 2] = voxel_features[:, :, 2] - (
                        coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

        # If the coordinates are absolute, splice the parts directly.
        if self.use_absolute_xyz:
            features = [voxel_features, f_cluster, f_center]
        # Otherwise, convert the voxel_features to 3D coordinates and then stitch the parts together.
        else:
            features = [voxel_features[..., 3:], f_cluster, f_center]

        # use distance information
        if self.with_distance:
            # In torch.norm function, the first 2 indicates solving L2 parametrization,
            # and the second 2 indicates solving parametrization in the third dimension.
            points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
            features.append(points_dist)
        # splice features in the last dimension
        features = torch.cat(features, dim=-1)

        # maximum number of point clouds in each pillar
        voxel_count = features.shape[1]
        
        # get the mask dimension
        # The mask specifies the data that should be retained in each pillar.
        mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)

        # up-dimensioning the mask
        mask = torch.unsqueeze(mask, -1).type_as(voxel_features)

        # set all features of the populated data in features to 0
        features *= mask

        for pfn in self.pfn_layers:
            features = pfn(features)

        # abstract a 64-dimensional feature in each pillar
        features = features.squeeze()
        batch_dict['pillar_features'] = features
        return batch_dict


class PillarVFE(VFETemplate):
    """
    model_cfg: NAME: PillarVFE
                     WITH_DISTANCE: False
                     USE_ABSLOTE_XYZ: True
                     NUM_FILTERS: [64]
    num_point_features: 4
    voxel_size: [0.16 0.16 4]
    POINT_CLOUD_RANGE: []
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                    last_layer=(i >= len(num_filters) - 2))
            )
        # add linear layers to increase the number of features
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.interral = interRAL(64)    # set the channel number of interRAL

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Args:
            actual_num: number of actual points per voxel
            max_num: the maximum number of voxel points
        Returns:
            paddings_indicator: Determine whether the data in the pillar is the real data or the filled value 0
        """

        # Extending a dimension
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        if 'voxels' in batch_dict:
            voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
            # Summing all point clouds in each pillar.
            # if keepdim=True is set, the original dimension information will be kept.
            # Divide the summation information by the number of points in each point cloud to get the average of all point clouds in each pillar.
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            
            # Subtract the average value of the corresponding pillar from each point cloud data to get the difference.
            f_cluster = voxel_features[:, :, :3] - points_mean
            
            # Resume the null data for each point cloud to the centroid offset of this pillar coordinate.
            f_center = torch.zeros_like(voxel_features[:, :, :3])

            # The coordinates (coords) of each grid point multiplied by the length and width of each pillar,
            # then we can obtain the actual length and width of the point cloud data (in m).
            # Add half of the length and width of each pillar to obtain the centroid coordinates of each pillar.
            # Subtract the centroid coordinates of the corresponding pillar from the x, y, and z of each point,
            # then we get the offset from each point to the centroid of the corresponding each point.
            f_center[:, :, 0] = voxel_features[:, :, 0] - (
                            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (
                            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (
                            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

            # If the coordinates are absolute, splice the parts directly.
            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            # Otherwise, convert the voxel_features to 3D coordinates and then stitch the parts together.
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]

            # use distance information
            if self.with_distance:
                # In torch.norm function, the first 2 indicates solving L2 parametrization,
                # and the second 2 indicates solving parametrization in the third dimension.
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            # splice features in the last dimension
            features = torch.cat(features, dim=-1)

            # maximum number of point clouds in each pillar
            voxel_count = features.shape[1]
            
            # get the mask dimension
            # The mask specifies the data that should be retained in each pillar.
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)

            # up-dimensioning the mask
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)

            # set all features of the populated data in features to 0
            features *= mask

            for pfn in self.pfn_layers:
                features = pfn(features)

            # abstract a 64-dimensional feature in each pillar
            features = features.squeeze()
            batch_dict['pillar_features'] = features

        else:
            # Process the information of different modalities in sequence and generate the results.
            lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
            radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
            lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
            radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
            radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

            lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
            radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
            lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
            radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)


            if self.use_absolute_xyz:
                lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            else:
                lidar_features = [lidar_voxel_features[..., 3:], lidar_f_cluster, lidar_f_center]
            if self.use_absolute_xyz:
                radar_features = [radar_voxel_features[..., :4], radar_f_cluster, radar_f_center] # 这里给去掉了
            else:
                radar_features = [radar_voxel_features[..., 3:], radar_f_cluster, radar_f_center]


            if self.with_distance:
                lidar_points_dist = torch.norm(lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                lidar_features.append(lidar_points_dist)
            lidar_features = torch.cat(lidar_features, dim=-1)
            if self.with_distance:
                radar_points_dist = torch.norm(radar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                radar_features.append(radar_points_dist)
            radar_features = torch.cat(radar_features, dim=-1)

            lidar_voxel_count = lidar_features.shape[1]
            radar_voxel_count = radar_features.shape[1]
            lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
            radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
            lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
            radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
            lidar_features *= lidar_mask
            radar_features *= radar_mask
            
            # for pfn in self.pfn_layers:
            #     lidar_features = pfn(lidar_features)
            # lidar_features = lidar_features.squeeze()
            # for pfn in self.pfn_layers:
            #     radar_features = pfn(radar_features)
            # radar_features = radar_features.squeeze()
            # 这里速度其他变量给去掉了
            # safusionlayer2
            
            lidar_features_output = self.interral(lidar_features, radar_features) # N,M,C
            radar_features_output = self.interral(radar_features, lidar_features)
            
            # 此处放入Top-K,这里并没有放top-k
            lidar_features = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            radar_features = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])

            batch_dict['lidar_pillar_features'] = lidar_features
            batch_dict['radar_pillar_features'] = radar_features
        
        return batch_dict

class PillarVFE_velocity(VFETemplate):
    """
    model_cfg: NAME: PillarVFE
                     WITH_DISTANCE: False
                     USE_ABSLOTE_XYZ: True
                     NUM_FILTERS: [64]
    num_point_features: 4
    voxel_size: [0.16 0.16 4]
    POINT_CLOUD_RANGE: []
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                    last_layer=(i >= len(num_filters) - 2))
            )
        # add linear layers to increase the number of features
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.interral = interRAL(64)    # set the channel number of interRAL
        
        pfn_layers = []
        # pfn_layers_multi = []
        num_filters[0]=13
        num_filters[1]=10
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_vel(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
            # pfn_layers_multi.append(
            #     PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            # )
        self.encoder_radar = nn.ModuleList(pfn_layers) #  14->10
       

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Args:
            actual_num: number of actual points per voxel
            max_num: the maximum number of voxel points
        Returns:
            paddings_indicator: Determine whether the data in the pillar is the real data or the filled value 0
        """

        # Extending a dimension
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        if 'voxels' in batch_dict:
            voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
            # Summing all point clouds in each pillar.
            # if keepdim=True is set, the original dimension information will be kept.
            # Divide the summation information by the number of points in each point cloud to get the average of all point clouds in each pillar.
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            
            # Subtract the average value of the corresponding pillar from each point cloud data to get the difference.
            f_cluster = voxel_features[:, :, :3] - points_mean
            
            # Resume the null data for each point cloud to the centroid offset of this pillar coordinate.
            f_center = torch.zeros_like(voxel_features[:, :, :3])

            # The coordinates (coords) of each grid point multiplied by the length and width of each pillar,
            # then we can obtain the actual length and width of the point cloud data (in m).
            # Add half of the length and width of each pillar to obtain the centroid coordinates of each pillar.
            # Subtract the centroid coordinates of the corresponding pillar from the x, y, and z of each point,
            # then we get the offset from each point to the centroid of the corresponding each point.
            f_center[:, :, 0] = voxel_features[:, :, 0] - (
                            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (
                            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (
                            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

            # If the coordinates are absolute, splice the parts directly.
            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            # Otherwise, convert the voxel_features to 3D coordinates and then stitch the parts together.
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]

            # use distance information
            if self.with_distance:
                # In torch.norm function, the first 2 indicates solving L2 parametrization,
                # and the second 2 indicates solving parametrization in the third dimension.
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            # splice features in the last dimension
            features = torch.cat(features, dim=-1)

            # maximum number of point clouds in each pillar
            voxel_count = features.shape[1]
            
            # get the mask dimension
            # The mask specifies the data that should be retained in each pillar.
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)

            # up-dimensioning the mask
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)

            # set all features of the populated data in features to 0
            features *= mask

            for pfn in self.pfn_layers:
                features = pfn(features)

            # abstract a 64-dimensional feature in each pillar
            features = features.squeeze()
            batch_dict['pillar_features'] = features

        else:
            # Process the information of different modalities in sequence and generate the results.
            lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
            radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
            lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
            radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
            radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

            lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
            radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
            lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
            radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)


            if self.use_absolute_xyz:
                lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            else:
                lidar_features = [lidar_voxel_features[..., 3:], lidar_f_cluster, lidar_f_center]
            if self.use_absolute_xyz:
                radar_features = [radar_voxel_features[..., :], radar_f_cluster, radar_f_center] # 这里给去掉了
            else:
                radar_features = [radar_voxel_features[..., 3:], radar_f_cluster, radar_f_center]


            if self.with_distance:
                lidar_points_dist = torch.norm(lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                lidar_features.append(lidar_points_dist)
            lidar_features = torch.cat(lidar_features, dim=-1)
            if self.with_distance:
                radar_points_dist = torch.norm(radar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                radar_features.append(radar_points_dist)
            radar_features = torch.cat(radar_features, dim=-1)

            lidar_voxel_count = lidar_features.shape[1]
            radar_voxel_count = radar_features.shape[1]
            lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
            radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
            lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
            radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
            lidar_features *= lidar_mask
            radar_features *= radar_mask

            # for pfn in self.pfn_layers:
            #     lidar_features = pfn(lidar_features)
            # lidar_features = lidar_features.squeeze()
            # for pfn in self.pfn_layers:
            #     radar_features = pfn(radar_features)
            # radar_features = radar_features.squeeze()
            # 这里速度其他变量给去掉了
            #   ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']  7个
            
            # safusionlayer2
            #  对radar加上一个特征提取
            # add_features_to_map = radar_features[:, :, 3:6]
            
            # add_features_to_map = torch.max(add_features_to_map, dim=1, keepdim=True)[0]
            # add_features_to_map = add_features_to_map.squeeze()
            # batch_dict['add_features_to_map'] = add_features_to_map
            
            radar_features = self.encoder_radar[0](radar_features)   # 后面是10
            
            lidar_features_output = self.interral(lidar_features, radar_features)
            radar_features_output = self.interral(radar_features, lidar_features)
            lidar_features = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            radar_features = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])

            batch_dict['lidar_pillar_features'] = lidar_features
            batch_dict['radar_pillar_features'] = radar_features
        
        return batch_dict

class PillarVFE_randanet(VFETemplate):
    """
    model_cfg: NAME: PillarVFE
                     WITH_DISTANCE: False
                     USE_ABSLOTE_XYZ: True
                     NUM_FILTERS: [64]
    num_point_features: 4
    voxel_size: [0.16 0.16 4]
    POINT_CLOUD_RANGE: []
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                    last_layer=(i >= len(num_filters) - 2))
            )
        # add linear layers to increase the number of features
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.interral = interRAL_other(64)    # set the channel number of interRAL
        
        pfn_layers = []
        # pfn_layers_multi = []
        # num_filters[0]=4 # 'rcs', 'v_r', 'v_r_comp', 'time'
        num_filters[0]= 4
        num_filters[1]=64
        
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_vel(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
            # pfn_layers_multi.append(
            #     PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            # )
        self.encoder_radar = nn.ModuleList(pfn_layers) #  14->10
        
        self.score_fn = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Softmax(dim=-2)
        )
       

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Args:
            actual_num: number of actual points per voxel
            max_num: the maximum number of voxel points
        Returns:
            paddings_indicator: Determine whether the data in the pillar is the real data or the filled value 0
        """

        # Extending a dimension
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        if 'voxels' in batch_dict:
            voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
            # Summing all point clouds in each pillar.
            # if keepdim=True is set, the original dimension information will be kept.
            # Divide the summation information by the number of points in each point cloud to get the average of all point clouds in each pillar.
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            
            # Subtract the average value of the corresponding pillar from each point cloud data to get the difference.
            f_cluster = voxel_features[:, :, :3] - points_mean
            
            # Resume the null data for each point cloud to the centroid offset of this pillar coordinate.
            f_center = torch.zeros_like(voxel_features[:, :, :3])

            # The coordinates (coords) of each grid point multiplied by the length and width of each pillar,
            # then we can obtain the actual length and width of the point cloud data (in m).
            # Add half of the length and width of each pillar to obtain the centroid coordinates of each pillar.
            # Subtract the centroid coordinates of the corresponding pillar from the x, y, and z of each point,
            # then we get the offset from each point to the centroid of the corresponding each point.
            f_center[:, :, 0] = voxel_features[:, :, 0] - (
                            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (
                            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (
                            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

            # If the coordinates are absolute, splice the parts directly.
            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            # Otherwise, convert the voxel_features to 3D coordinates and then stitch the parts together.
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]

            # use distance information
            if self.with_distance:
                # In torch.norm function, the first 2 indicates solving L2 parametrization,
                # and the second 2 indicates solving parametrization in the third dimension.
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            # splice features in the last dimension
            features = torch.cat(features, dim=-1)

            # maximum number of point clouds in each pillar
            voxel_count = features.shape[1]
            
            # get the mask dimension
            # The mask specifies the data that should be retained in each pillar.
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)

            # up-dimensioning the mask
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)

            # set all features of the populated data in features to 0
            features *= mask

            for pfn in self.pfn_layers:
                features = pfn(features)

            # abstract a 64-dimensional feature in each pillar
            features = features.squeeze()
            batch_dict['pillar_features'] = features

        else:
            # Process the information of different modalities in sequence and generate the results.
            lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
            radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
            lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
            radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
            radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

            lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
            radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
            lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
            radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)


            if self.use_absolute_xyz:
                lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            else:
                lidar_features = [lidar_voxel_features[..., 3:], lidar_f_cluster, lidar_f_center]
            if self.use_absolute_xyz:
                radar_features = [radar_voxel_features[..., :], radar_f_cluster, radar_f_center] # 这里给去掉了
            else:
                radar_features = [radar_voxel_features[..., 3:], radar_f_cluster, radar_f_center]


            if self.with_distance:
                lidar_points_dist = torch.norm(lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                lidar_features.append(lidar_points_dist)
            lidar_features = torch.cat(lidar_features, dim=-1)
            if self.with_distance:
                radar_points_dist = torch.norm(radar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                radar_features.append(radar_points_dist)
            radar_features = torch.cat(radar_features, dim=-1)

            lidar_voxel_count = lidar_features.shape[1]
            radar_voxel_count = radar_features.shape[1]
            lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
            radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
            lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
            radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
            lidar_features *= lidar_mask
            radar_features *= radar_mask

            # for pfn in self.pfn_layers:
            #     lidar_features = pfn(lidar_features)
            # lidar_features = lidar_features.squeeze()
            # for pfn in self.pfn_layers:
            #     radar_features = pfn(radar_features)
            # radar_features = radar_features.squeeze()
            # 这里速度其他变量给去掉了
            #   ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']  7个
            
            # safusionlayer2
            #  对radar加上一个特征提取
            add_features_to_map = radar_features[:, :, 3:7] # 'rcs', 'v_r', 'v_r_comp', 'time'
            # add_features_to_map = radar_features[:, :, 3].unsqueeze(dim=-1) # RCS
                
            # add_features_to_map = torch.max(add_features_to_map, dim=1, keepdim=True)[0]
            # add_features_to_map = add_features_to_map.squeeze()
            # batch_dict['add_features_to_map'] = add_features_to_map
            
            # radar_features = self.encoder_radar[0](radar_features)   # 后面是10
            radar_features = radar_features[:,:,[0,1,2,6,7,8,9,10,11,12]]  # 这里感觉还是加上了时间的维度
            
            add_features_to_map = self.encoder_radar[0](add_features_to_map)   # 得到64维度的额外编码 # B,N,C
            
            add_features_to_map = self.score_fn(add_features_to_map).permute(0,2,1).contiguous()  # B,C,N
            
            
            
            
            lidar_features_output = self.interral(lidar_features, radar_features)
            radar_features_output = self.interral(radar_features, lidar_features) # B,C，N
            
            # 再这里额外对速度做了一个处理，类似于Randanet中软注意力机制
            
            radar_features_output = add_features_to_map*radar_features_output
            radar_features_output = torch.max(radar_features_output, dim=2, keepdim=True)[0]
            
            lidar_features_output = torch.max(lidar_features_output, dim=2, keepdim=True)[0]
            
            lidar_features = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            radar_features = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])

            batch_dict['lidar_pillar_features'] = lidar_features
            batch_dict['radar_pillar_features'] = radar_features
        
        return batch_dict


class PillarVFE_randanet_new(VFETemplate):
    """
    model_cfg: NAME: PillarVFE
                     WITH_DISTANCE: False
                     USE_ABSLOTE_XYZ: True
                     NUM_FILTERS: [64]
    num_point_features: 4
    voxel_size: [0.16 0.16 4]
    POINT_CLOUD_RANGE: []
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                    last_layer=(i >= len(num_filters) - 2))
            )
        # add linear layers to increase the number of features
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.interral = interRAL_other(64)    # set the channel number of interRAL
        
        pfn_layers = []
        # pfn_layers_multi = []
        # num_filters[0]=4 # 'rcs', 'v_r', 'v_r_comp', 'time'
        num_filters[0]= 4
        num_filters[1]=64
        
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_vel(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
            # pfn_layers_multi.append(
            #     PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            # )
        self.encoder_radar = nn.ModuleList(pfn_layers) #  14->10
        
        
        num_filters[0]= 13
        num_filters[1]=10
        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_vel(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
            # pfn_layers_multi.append(
            #     PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            # )
        self.encoder_radar2= nn.ModuleList(pfn_layers) #  14->10
        
        
        self.score_fn = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Softmax(dim=-2)
        )
       

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Args:
            actual_num: number of actual points per voxel
            max_num: the maximum number of voxel points
        Returns:
            paddings_indicator: Determine whether the data in the pillar is the real data or the filled value 0
        """

        # Extending a dimension
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        if 'voxels' in batch_dict:
            voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
            # Summing all point clouds in each pillar.
            # if keepdim=True is set, the original dimension information will be kept.
            # Divide the summation information by the number of points in each point cloud to get the average of all point clouds in each pillar.
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            
            # Subtract the average value of the corresponding pillar from each point cloud data to get the difference.
            f_cluster = voxel_features[:, :, :3] - points_mean
            
            # Resume the null data for each point cloud to the centroid offset of this pillar coordinate.
            f_center = torch.zeros_like(voxel_features[:, :, :3])

            # The coordinates (coords) of each grid point multiplied by the length and width of each pillar,
            # then we can obtain the actual length and width of the point cloud data (in m).
            # Add half of the length and width of each pillar to obtain the centroid coordinates of each pillar.
            # Subtract the centroid coordinates of the corresponding pillar from the x, y, and z of each point,
            # then we get the offset from each point to the centroid of the corresponding each point.
            f_center[:, :, 0] = voxel_features[:, :, 0] - (
                            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (
                            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (
                            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

            # If the coordinates are absolute, splice the parts directly.
            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            # Otherwise, convert the voxel_features to 3D coordinates and then stitch the parts together.
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]

            # use distance information
            if self.with_distance:
                # In torch.norm function, the first 2 indicates solving L2 parametrization,
                # and the second 2 indicates solving parametrization in the third dimension.
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            # splice features in the last dimension
            features = torch.cat(features, dim=-1)

            # maximum number of point clouds in each pillar
            voxel_count = features.shape[1]
            
            # get the mask dimension
            # The mask specifies the data that should be retained in each pillar.
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)

            # up-dimensioning the mask
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)

            # set all features of the populated data in features to 0
            features *= mask

            for pfn in self.pfn_layers:
                features = pfn(features)

            # abstract a 64-dimensional feature in each pillar
            features = features.squeeze()
            batch_dict['pillar_features'] = features

        else:
            # Process the information of different modalities in sequence and generate the results.
            lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
            radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
            lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
            radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
            radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

            lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
            radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
            lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
            radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)


            if self.use_absolute_xyz:
                lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            else:
                lidar_features = [lidar_voxel_features[..., 3:], lidar_f_cluster, lidar_f_center]
            if self.use_absolute_xyz:
                radar_features = [radar_voxel_features[..., :], radar_f_cluster, radar_f_center] # 这里给去掉了
            else:
                radar_features = [radar_voxel_features[..., 3:], radar_f_cluster, radar_f_center]


            if self.with_distance:
                lidar_points_dist = torch.norm(lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                lidar_features.append(lidar_points_dist)
            lidar_features = torch.cat(lidar_features, dim=-1)
            if self.with_distance:
                radar_points_dist = torch.norm(radar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                radar_features.append(radar_points_dist)
            radar_features = torch.cat(radar_features, dim=-1)

            lidar_voxel_count = lidar_features.shape[1]
            radar_voxel_count = radar_features.shape[1]
            lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
            radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
            lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
            radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
            lidar_features *= lidar_mask
            radar_features *= radar_mask

            # for pfn in self.pfn_layers:
            #     lidar_features = pfn(lidar_features)
            # lidar_features = lidar_features.squeeze()
            # for pfn in self.pfn_layers:
            #     radar_features = pfn(radar_features)
            # radar_features = radar_features.squeeze()
            # 这里速度其他变量给去掉了
            #   ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']  7个
            
            # safusionlayer2
            #  对radar加上一个特征提取
            add_features_to_map = radar_features[:, :, 3:7] # 'rcs', 'v_r', 'v_r_comp', 'time'
            # add_features_to_map = radar_features[:, :, 3].unsqueeze(dim=-1) # RCS
                
            # add_features_to_map = torch.max(add_features_to_map, dim=1, keepdim=True)[0]
            # add_features_to_map = add_features_to_map.squeeze()
            # batch_dict['add_features_to_map'] = add_features_to_map
            
            
            # 8.26 这个我屏蔽了
            # radar_features = radar_features[:,:,[0,1,2,6,7,8,9,10,11,12]]  # 这里感觉还是加上了时间的维度
            radar_features = self.encoder_radar2[0](radar_features)   # 后面是10
            
            add_features_to_map = self.encoder_radar[0](add_features_to_map)   # 得到64维度的额外编码 # B,N,C
            
            add_features_to_map = self.score_fn(add_features_to_map).permute(0,2,1).contiguous()  # B,C,N
            
            
            
            
            lidar_features_output = self.interral(lidar_features, radar_features)
            radar_features_output = self.interral(radar_features, lidar_features) # B,C，N
            
            # 再这里额外对速度做了一个处理，类似于Randanet中软注意力机制
            
            radar_features_output = add_features_to_map*radar_features_output
            radar_features_output = torch.max(radar_features_output, dim=2, keepdim=True)[0]
            
            lidar_features_output = torch.max(lidar_features_output, dim=2, keepdim=True)[0]
            
            lidar_features = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            radar_features = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])

            batch_dict['lidar_pillar_features'] = lidar_features
            batch_dict['radar_pillar_features'] = radar_features
        
        return batch_dict


class PillarVFE_randanet_radar_lidar(VFETemplate):
    """
    model_cfg: NAME: PillarVFE
                     WITH_DISTANCE: False
                     USE_ABSLOTE_XYZ: True
                     NUM_FILTERS: [64]
    num_point_features: 4
    voxel_size: [0.16 0.16 4]
    POINT_CLOUD_RANGE: []
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                    last_layer=(i >= len(num_filters) - 2))
            )
        # add linear layers to increase the number of features
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.interral = interRAL_other(64)    # set the channel number of interRAL
        self.interral_lidar = interRAL_other_lidar_voc(64)    # set the channel number of interRAL
        
        
        pfn_layers = []
        # pfn_layers_multi = []
        num_filters[0]= 2 # 4
        num_filters[1]=64
        
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_vel(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
            # pfn_layers_multi.append(
            #     PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            # )
        self.encoder_radar = nn.ModuleList(pfn_layers) #  14->10
        
        self.score_fn = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Softmax(dim=-2)
        )
        
        self.score_fn_radar = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Softmax(dim=-2)
        )

        self.pointnet1_test = PointNet(in_channels=10,
                                  out_channels=64,
                                  use_norm=True,
                                  last_layer=True)
        
        self.pointnet2_test = PointNet(in_channels=10,
                                  out_channels=64,
                                  use_norm=True,
                                  last_layer=True)
        
        # self.pointnet1 = PointNet(in_channels=1,
        #                           out_channels=32,
        #                           use_norm=True,
        #                           last_layer=True)
        
        # self.pointnet2 = PointNet(in_channels=1,
        #                           out_channels=32,
        #                           use_norm=True,
        #                           last_layer=True)
        
        # self.interral_h_i = interRAL_Lidar_other(64)    # set the channel number of interRAL

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Args:
            actual_num: number of actual points per voxel
            max_num: the maximum number of voxel points
        Returns:
            paddings_indicator: Determine whether the data in the pillar is the real data or the filled value 0
        """

        # Extending a dimension
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        if 'voxels' in batch_dict:
            voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
            # Summing all point clouds in each pillar.
            # if keepdim=True is set, the original dimension information will be kept.
            # Divide the summation information by the number of points in each point cloud to get the average of all point clouds in each pillar.
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            
            # Subtract the average value of the corresponding pillar from each point cloud data to get the difference.
            f_cluster = voxel_features[:, :, :3] - points_mean
            
            # Resume the null data for each point cloud to the centroid offset of this pillar coordinate.
            f_center = torch.zeros_like(voxel_features[:, :, :3])

            # The coordinates (coords) of each grid point multiplied by the length and width of each pillar,
            # then we can obtain the actual length and width of the point cloud data (in m).
            # Add half of the length and width of each pillar to obtain the centroid coordinates of each pillar.
            # Subtract the centroid coordinates of the corresponding pillar from the x, y, and z of each point,
            # then we get the offset from each point to the centroid of the corresponding each point.
            f_center[:, :, 0] = voxel_features[:, :, 0] - (
                            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (
                            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (
                            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

            # If the coordinates are absolute, splice the parts directly.
            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            # Otherwise, convert the voxel_features to 3D coordinates and then stitch the parts together.
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]

            # use distance information
            if self.with_distance:
                # In torch.norm function, the first 2 indicates solving L2 parametrization,
                # and the second 2 indicates solving parametrization in the third dimension.
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            # splice features in the last dimension
            features = torch.cat(features, dim=-1)

            # maximum number of point clouds in each pillar
            voxel_count = features.shape[1]
            
            # get the mask dimension
            # The mask specifies the data that should be retained in each pillar.
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)

            # up-dimensioning the mask
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)

            # set all features of the populated data in features to 0
            features *= mask

            for pfn in self.pfn_layers:
                features = pfn(features)

            # abstract a 64-dimensional feature in each pillar
            features = features.squeeze()
            batch_dict['pillar_features'] = features

        else:
            # Process the information of different modalities in sequence and generate the results.
            lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
            radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
            lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
            radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
            radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

            lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
            radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
            lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
            radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)


            if self.use_absolute_xyz:
                lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            else:
                lidar_features = [lidar_voxel_features[..., 3:], lidar_f_cluster, lidar_f_center]
            if self.use_absolute_xyz:
                radar_features = [radar_voxel_features[..., :], radar_f_cluster, radar_f_center] # 这里给去掉了
            else:
                radar_features = [radar_voxel_features[..., 3:], radar_f_cluster, radar_f_center]


            if self.with_distance:
                lidar_points_dist = torch.norm(lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                lidar_features.append(lidar_points_dist)
            lidar_features = torch.cat(lidar_features, dim=-1)
            if self.with_distance:
                radar_points_dist = torch.norm(radar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                radar_features.append(radar_points_dist)
            radar_features = torch.cat(radar_features, dim=-1)

            lidar_voxel_count = lidar_features.shape[1]
            radar_voxel_count = radar_features.shape[1]
            lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
            radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
            lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
            radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
            lidar_features *= lidar_mask
            radar_features *= radar_mask

            # for pfn in self.pfn_layers:
            #     lidar_features = pfn(lidar_features)
            # lidar_features = lidar_features.squeeze()
            # for pfn in self.pfn_layers:
            #     radar_features = pfn(radar_features)
            # radar_features = radar_features.squeeze()
            # 这里速度其他变量给去掉了
            #   ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']  7个
            
            # safusionlayer2
            #  对radar加上一个特征提取
            add_features_to_map = radar_features[:, :, [3,6]]# 3:7 .unsqueeze(dim=-1)  # 3:7
            
            # add_features_to_map = torch.max(add_features_to_map, dim=1, keepdim=True)[0]
            # add_features_to_map = add_features_to_map.squeeze()
            # batch_dict['add_features_to_map'] = add_features_to_map
            
            # radar_features = self.encoder_radar[0](radar_features)   # 后面是10
            radar_features = radar_features[:,:,[0,1,2,6,7,8,9,10,11,12]]  # 这里感觉还是加上了时间的维度
            
            add_features_to_map = self.encoder_radar[0](add_features_to_map)   # 得到64维度的额外编码 # B,N,C  
            
            add_features_to_map = self.score_fn(add_features_to_map).permute(0,2,1).contiguous()  # B,C,N
            
            # 如果是lidar额外点
            # 取出高度
            '''lidar_features_height = lidar_features[:,:,2].unsqueeze(-1).contiguous()
            lidar_features_insentiy = lidar_features[:,:,3].unsqueeze(-1).contiguous()
            lidar_features_height = self.pointnet1(lidar_features_height)
            lidar_features_insentiy  = self.pointnet2(lidar_features_insentiy)
            lidar_other_single_feature = torch.cat([lidar_features_height,lidar_features_insentiy],dim=-1)'''
            
            
            
            
            # 将interfusion屏蔽掉，单独特征提取看看
            # lidar_features_output= self.pointnet1_test(lidar_features).permute(0,2,1).contiguous()
            # radar_features_output  = self.pointnet2_test(radar_features).permute(0,2,1).contiguous()       
            
            
            lidar_features_output = self.interral(lidar_features, radar_features)
            radar_features_output = self.interral(radar_features, lidar_features) # B,C，N
            
            # 7-5 测试加的，使用lidar得高度进行加权
            # radar_features_output = self.interral_h_i(radar_features_output.permute(0,2,1).contiguous(),lidar_other_single_feature)
            
            # 对radar来个自注意力机制  这一行其实有点怪怪的，应该直接加权就行了
            # # 8.3 这个后面两行得打开，这是为了做R-L得实验得 
            
            radar_features_output = radar_features_output + radar_features_output*self.score_fn_radar((radar_features_output+add_features_to_map).permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
            radar_features_output = add_features_to_map*radar_features_output  # 再这里额外对速度做了一个处理，类似于Randanet中软注意力机制  这里面可以再用相加的符号
            
            radar_features_output = torch.max(radar_features_output, dim=2, keepdim=True)[0]
            
            # 用速度其他属性对lidar进行加权
            # 8.2 这个后面得打开，这是为了做R-R得实验得
            
            lidar_features_output = self.interral_lidar(lidar_features_output.permute(0,2,1).contiguous() , add_features_to_map.permute(0,2,1).contiguous())
          
            lidar_features_output = torch.max(lidar_features_output, dim=2, keepdim=True)[0]
            
            
            
            lidar_features = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            radar_features = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])

            batch_dict['lidar_pillar_features'] = lidar_features
            batch_dict['radar_pillar_features'] = radar_features
        
        return batch_dict



class PillarVFE_randanet_new_radar_lidar(VFETemplate):
    """
    model_cfg: NAME: PillarVFE
                     WITH_DISTANCE: False
                     USE_ABSLOTE_XYZ: True
                     NUM_FILTERS: [64]
    num_point_features: 4
    voxel_size: [0.16 0.16 4]
    POINT_CLOUD_RANGE: []
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                    last_layer=(i >= len(num_filters) - 2))
            )
        # add linear layers to increase the number of features
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.interral = interRAL_other(64)    # set the channel number of interRAL
        self.interral_lidar = interRAL_other_lidar_voc(64)    # set the channel number of interRAL
        
        
        pfn_layers = []
        # pfn_layers_multi = []
        num_filters[0]= 4 # 4
        num_filters[1]=64
        
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_vel(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
            # pfn_layers_multi.append(
            #     PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            # )
        self.encoder_radar = nn.ModuleList(pfn_layers) #  14->10
        
        
        
        
        # pfn_layers_multi = []
        num_filters[0]= 13 # 4
        num_filters[1]=10
        pfn_layers= []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_vel(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
            # pfn_layers_multi.append(
            #     PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            # )
        self.encoder_radar2 = nn.ModuleList(pfn_layers) #  14->10
        
        
        self.score_fn = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Softmax(dim=-2)
        )
        
        self.score_fn_radar = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Softmax(dim=-2)
        )

        self.pointnet1_test = PointNet(in_channels=10,
                                  out_channels=64,
                                  use_norm=True,
                                  last_layer=True)
        
        self.pointnet2_test = PointNet(in_channels=10,
                                  out_channels=64,
                                  use_norm=True,
                                  last_layer=True)
        
        # self.pointnet1 = PointNet(in_channels=1,
        #                           out_channels=32,
        #                           use_norm=True,
        #                           last_layer=True)
        
        # self.pointnet2 = PointNet(in_channels=1,
        #                           out_channels=32,
        #                           use_norm=True,
        #                           last_layer=True)
        
        # self.interral_h_i = interRAL_Lidar_other(64)    # set the channel number of interRAL

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Args:
            actual_num: number of actual points per voxel
            max_num: the maximum number of voxel points
        Returns:
            paddings_indicator: Determine whether the data in the pillar is the real data or the filled value 0
        """

        # Extending a dimension
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        if 'voxels' in batch_dict:
            voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
            # Summing all point clouds in each pillar.
            # if keepdim=True is set, the original dimension information will be kept.
            # Divide the summation information by the number of points in each point cloud to get the average of all point clouds in each pillar.
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            
            # Subtract the average value of the corresponding pillar from each point cloud data to get the difference.
            f_cluster = voxel_features[:, :, :3] - points_mean
            
            # Resume the null data for each point cloud to the centroid offset of this pillar coordinate.
            f_center = torch.zeros_like(voxel_features[:, :, :3])

            # The coordinates (coords) of each grid point multiplied by the length and width of each pillar,
            # then we can obtain the actual length and width of the point cloud data (in m).
            # Add half of the length and width of each pillar to obtain the centroid coordinates of each pillar.
            # Subtract the centroid coordinates of the corresponding pillar from the x, y, and z of each point,
            # then we get the offset from each point to the centroid of the corresponding each point.
            f_center[:, :, 0] = voxel_features[:, :, 0] - (
                            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (
                            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (
                            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

            # If the coordinates are absolute, splice the parts directly.
            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            # Otherwise, convert the voxel_features to 3D coordinates and then stitch the parts together.
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]

            # use distance information
            if self.with_distance:
                # In torch.norm function, the first 2 indicates solving L2 parametrization,
                # and the second 2 indicates solving parametrization in the third dimension.
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            # splice features in the last dimension
            features = torch.cat(features, dim=-1)

            # maximum number of point clouds in each pillar
            voxel_count = features.shape[1]
            
            # get the mask dimension
            # The mask specifies the data that should be retained in each pillar.
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)

            # up-dimensioning the mask
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)

            # set all features of the populated data in features to 0
            features *= mask

            for pfn in self.pfn_layers:
                features = pfn(features)

            # abstract a 64-dimensional feature in each pillar
            features = features.squeeze()
            batch_dict['pillar_features'] = features

        else:
            # Process the information of different modalities in sequence and generate the results.
            lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
            radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
            lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
            radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
            radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

            lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
            radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
            lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
            radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)


            if self.use_absolute_xyz:
                lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            else:
                lidar_features = [lidar_voxel_features[..., 3:], lidar_f_cluster, lidar_f_center]
            if self.use_absolute_xyz:
                radar_features = [radar_voxel_features[..., :], radar_f_cluster, radar_f_center] # 这里给去掉了
            else:
                radar_features = [radar_voxel_features[..., 3:], radar_f_cluster, radar_f_center]


            if self.with_distance:
                lidar_points_dist = torch.norm(lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                lidar_features.append(lidar_points_dist)
            lidar_features = torch.cat(lidar_features, dim=-1)
            if self.with_distance:
                radar_points_dist = torch.norm(radar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                radar_features.append(radar_points_dist)
            radar_features = torch.cat(radar_features, dim=-1)

            lidar_voxel_count = lidar_features.shape[1]
            radar_voxel_count = radar_features.shape[1]
            lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
            radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
            lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
            radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
            lidar_features *= lidar_mask
            radar_features *= radar_mask

            # for pfn in self.pfn_layers:
            #     lidar_features = pfn(lidar_features)
            # lidar_features = lidar_features.squeeze()
            # for pfn in self.pfn_layers:
            #     radar_features = pfn(radar_features)
            # radar_features = radar_features.squeeze()
            # 这里速度其他变量给去掉了
            #   ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']  7个
            
            # safusionlayer2
            #  对radar加上一个特征提取
            add_features_to_map = radar_features[:, :, 3:7]# 3:7 .unsqueeze(dim=-1)  # 3:7
            
            # add_features_to_map = torch.max(add_features_to_map, dim=1, keepdim=True)[0]
            # add_features_to_map = add_features_to_map.squeeze()
            # batch_dict['add_features_to_map'] = add_features_to_map
            
            radar_features = self.encoder_radar2[0](radar_features)   # 后面是10
            # radar_features = radar_features[:,:,[0,1,2,6,7,8,9,10,11,12]]  # 这里感觉还是加上了时间的维度
            
            add_features_to_map = self.encoder_radar[0](add_features_to_map)   # 得到64维度的额外编码 # B,N,C  
            
            add_features_to_map = self.score_fn(add_features_to_map).permute(0,2,1).contiguous()  # B,C,N
            
            # 如果是lidar额外点
            # 取出高度
            '''lidar_features_height = lidar_features[:,:,2].unsqueeze(-1).contiguous()
            lidar_features_insentiy = lidar_features[:,:,3].unsqueeze(-1).contiguous()
            lidar_features_height = self.pointnet1(lidar_features_height)
            lidar_features_insentiy  = self.pointnet2(lidar_features_insentiy)
            lidar_other_single_feature = torch.cat([lidar_features_height,lidar_features_insentiy],dim=-1)'''
            
            
            
            
            # 将interfusion屏蔽掉，单独特征提取看看
            # lidar_features_output= self.pointnet1_test(lidar_features).permute(0,2,1).contiguous()
            # radar_features_output  = self.pointnet2_test(radar_features).permute(0,2,1).contiguous()       
            
            
            lidar_features_output = self.interral(lidar_features, radar_features)
            radar_features_output = self.interral(radar_features, lidar_features) # B,C，N
            
            # 7-5 测试加的，使用lidar得高度进行加权
            # radar_features_output = self.interral_h_i(radar_features_output.permute(0,2,1).contiguous(),lidar_other_single_feature)
            
            # 对radar来个自注意力机制  这一行其实有点怪怪的，应该直接加权就行了
            # # 8.3 这个后面两行得打开，这是为了做R-L得实验得 
            
            radar_features_output = radar_features_output + radar_features_output*self.score_fn_radar((radar_features_output+add_features_to_map).permute(0,2,1).contiguous()).permute(0,2,1).contiguous()
            radar_features_output = add_features_to_map*radar_features_output  # 再这里额外对速度做了一个处理，类似于Randanet中软注意力机制  这里面可以再用相加的符号
            
            radar_features_output = torch.max(radar_features_output, dim=2, keepdim=True)[0]
            
            # 用速度其他属性对lidar进行加权
            # 8.2 这个后面得打开，这是为了做R-R得实验得
            
            lidar_features_output = self.interral_lidar(lidar_features_output.permute(0,2,1).contiguous() , add_features_to_map.permute(0,2,1).contiguous())
          
            lidar_features_output = torch.max(lidar_features_output, dim=2, keepdim=True)[0]
            
            
            
            lidar_features = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            radar_features = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])

            batch_dict['lidar_pillar_features'] = lidar_features
            batch_dict['radar_pillar_features'] = radar_features
        
        return batch_dict

class PointNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        """
        in_channels: 10
        out_channels: 64
        """
        super().__init__()
        
        self.last_vfe = last_layer # True
        self.use_norm = use_norm # True
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False) # 线性层 + BatchNorm
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)

        self.part = 50000

    def forward(self, inputs):
        """
        inputs:（31530，32，10)
        """
        if inputs.shape[0] > self.part:
            # nn.Linear performs randomly when batch size is too large
            num_parts = inputs.shape[0] // self.part
            part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
                               for num_part in range(num_parts+1)]
            x = torch.cat(part_linear_out, dim=0)
        else:
            # 线性层
            x = self.linear(inputs) # (31530,32,64)
        torch.backends.cudnn.enabled = False
        # BatchNorm1d层:(31530, 64, 32) --> (31530, 32, 64)
        # 这里之所以变换维度，是因为BatchNorm1d在通道维度上进行,对于图像来说默认模式为[N,C,H*W],通道在第二个维度上
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
        torch.backends.cudnn.enabled = True
        # 激活函数
        x = F.relu(x)
        # 按照维度取每个voxel中的最大值 --> (31530, 1, 64)
        # 这里的0是表示取数值，max的1表示索引
        # x_max = torch.max(x, dim=1, keepdim=True)[0]
        x_max = x

        if self.last_vfe:
            return x_max
        else:
            # torch的repeat在第几维度复制几遍
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            # 在最后一个维度上拼接 
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated



class PillarVFE_randanet_radar_lidar_our_attention(VFETemplate):
    """
    model_cfg: NAME: PillarVFE
                     WITH_DISTANCE: False
                     USE_ABSLOTE_XYZ: True
                     NUM_FILTERS: [64]
    num_point_features: 4
    voxel_size: [0.16 0.16 4]
    POINT_CLOUD_RANGE: []
    """

    def __init__(self, model_cfg, num_point_features, voxel_size, point_cloud_range):
        super().__init__(model_cfg=model_cfg)

        self.use_norm = self.model_cfg.USE_NORM
        self.with_distance = self.model_cfg.WITH_DISTANCE
        self.use_absolute_xyz = self.model_cfg.USE_ABSLOTE_XYZ
        num_point_features += 6 if self.use_absolute_xyz else 3
        if self.with_distance:
            num_point_features += 1

        self.num_filters = self.model_cfg.NUM_FILTERS
        assert len(self.num_filters) > 0
        num_filters = [num_point_features] + list(self.num_filters)

        pfn_layers = []
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer(in_filters, out_filters, self.use_norm,
                    last_layer=(i >= len(num_filters) - 2))
            )
        # add linear layers to increase the number of features
        self.pfn_layers = nn.ModuleList(pfn_layers)

        self.voxel_x = voxel_size[0]
        self.voxel_y = voxel_size[1]
        self.voxel_z = voxel_size[2]
        self.x_offset = self.voxel_x / 2 + point_cloud_range[0]
        self.y_offset = self.voxel_y / 2 + point_cloud_range[1]
        self.z_offset = self.voxel_z / 2 + point_cloud_range[2]
        self.interral = interRAL_other(64)    # set the channel number of interRAL
        self.interral_lidar = interRAL_other_lidar_voc(64)    # set the channel number of interRAL
        
        pfn_layers = []
        # pfn_layers_multi = []
        num_filters[0]=4
        num_filters[1]=64
        
        for i in range(len(num_filters) - 1):
            in_filters = num_filters[i]
            out_filters = num_filters[i + 1]
            pfn_layers.append(
                PFNLayer_vel(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            )
            # pfn_layers_multi.append(
            #     PFNLayer(in_filters, out_filters, self.use_norm, last_layer=(i >= len(num_filters) - 2))
            # )
        self.encoder_radar = nn.ModuleList(pfn_layers) #  14->10
        
        self.score_fn = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.Softmax(dim=-2)
        )
       

    def get_output_feature_dim(self):
        return self.num_filters[-1]

    def get_paddings_indicator(self, actual_num, max_num, axis=0):
        """
        Args:
            actual_num: number of actual points per voxel
            max_num: the maximum number of voxel points
        Returns:
            paddings_indicator: Determine whether the data in the pillar is the real data or the filled value 0
        """

        # Extending a dimension
        actual_num = torch.unsqueeze(actual_num, axis + 1)
        max_num_shape = [1] * len(actual_num.shape)
        max_num_shape[axis + 1] = -1
        max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(max_num_shape)
        paddings_indicator = actual_num.int() > max_num
        return paddings_indicator

    def forward(self, batch_dict, **kwargs):
        if 'voxels' in batch_dict:
            voxel_features, voxel_num_points, coords = batch_dict['voxels'], batch_dict['voxel_num_points'], batch_dict['voxel_coords']
            # Summing all point clouds in each pillar.
            # if keepdim=True is set, the original dimension information will be kept.
            # Divide the summation information by the number of points in each point cloud to get the average of all point clouds in each pillar.
            points_mean = voxel_features[:, :, :3].sum(dim=1, keepdim=True) / voxel_num_points.type_as(voxel_features).view(-1, 1, 1)
            
            # Subtract the average value of the corresponding pillar from each point cloud data to get the difference.
            f_cluster = voxel_features[:, :, :3] - points_mean
            
            # Resume the null data for each point cloud to the centroid offset of this pillar coordinate.
            f_center = torch.zeros_like(voxel_features[:, :, :3])

            # The coordinates (coords) of each grid point multiplied by the length and width of each pillar,
            # then we can obtain the actual length and width of the point cloud data (in m).
            # Add half of the length and width of each pillar to obtain the centroid coordinates of each pillar.
            # Subtract the centroid coordinates of the corresponding pillar from the x, y, and z of each point,
            # then we get the offset from each point to the centroid of the corresponding each point.
            f_center[:, :, 0] = voxel_features[:, :, 0] - (
                            coords[:, 3].to(voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            f_center[:, :, 1] = voxel_features[:, :, 1] - (
                            coords[:, 2].to(voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            f_center[:, :, 2] = voxel_features[:, :, 2] - (
                            coords[:, 1].to(voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)

            # If the coordinates are absolute, splice the parts directly.
            if self.use_absolute_xyz:
                features = [voxel_features, f_cluster, f_center]
            # Otherwise, convert the voxel_features to 3D coordinates and then stitch the parts together.
            else:
                features = [voxel_features[..., 3:], f_cluster, f_center]

            # use distance information
            if self.with_distance:
                # In torch.norm function, the first 2 indicates solving L2 parametrization,
                # and the second 2 indicates solving parametrization in the third dimension.
                points_dist = torch.norm(voxel_features[:, :, :3], 2, 2, keepdim=True)
                features.append(points_dist)
            # splice features in the last dimension
            features = torch.cat(features, dim=-1)

            # maximum number of point clouds in each pillar
            voxel_count = features.shape[1]
            
            # get the mask dimension
            # The mask specifies the data that should be retained in each pillar.
            mask = self.get_paddings_indicator(voxel_num_points, voxel_count, axis=0)

            # up-dimensioning the mask
            mask = torch.unsqueeze(mask, -1).type_as(voxel_features)

            # set all features of the populated data in features to 0
            features *= mask

            for pfn in self.pfn_layers:
                features = pfn(features)

            # abstract a 64-dimensional feature in each pillar
            features = features.squeeze()
            batch_dict['pillar_features'] = features

        else:
            # Process the information of different modalities in sequence and generate the results.
            lidar_voxel_features, lidar_voxel_num_points, lidar_coords = batch_dict['lidar_voxels'], batch_dict['lidar_voxel_num_points'], batch_dict['lidar_voxel_coords']
            radar_voxel_features, radar_voxel_num_points, radar_coords = batch_dict['radar_voxels'], batch_dict['radar_voxel_num_points'], batch_dict['radar_voxel_coords']
            lidar_points_mean = lidar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / lidar_voxel_num_points.type_as(lidar_voxel_features).view(-1, 1, 1)
            radar_points_mean = radar_voxel_features[:, :, :3].sum(dim=1, keepdim=True) / radar_voxel_num_points.type_as(radar_voxel_features).view(-1, 1, 1)
            lidar_f_cluster = lidar_voxel_features[:, :, :3] - lidar_points_mean
            radar_f_cluster = radar_voxel_features[:, :, :3] - radar_points_mean

            lidar_f_center = torch.zeros_like(lidar_voxel_features[:, :, :3])
            radar_f_center = torch.zeros_like(radar_voxel_features[:, :, :3])
            lidar_f_center[:, :, 0] = lidar_voxel_features[:, :, 0] - (lidar_coords[:, 3].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            lidar_f_center[:, :, 1] = lidar_voxel_features[:, :, 1] - (lidar_coords[:, 2].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            lidar_f_center[:, :, 2] = lidar_voxel_features[:, :, 2] - (lidar_coords[:, 1].to(lidar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)
            radar_f_center[:, :, 0] = radar_voxel_features[:, :, 0] - (radar_coords[:, 3].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_x + self.x_offset)
            radar_f_center[:, :, 1] = radar_voxel_features[:, :, 1] - (radar_coords[:, 2].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_y + self.y_offset)
            radar_f_center[:, :, 2] = radar_voxel_features[:, :, 2] - (radar_coords[:, 1].to(radar_voxel_features.dtype).unsqueeze(1) * self.voxel_z + self.z_offset)


            if self.use_absolute_xyz:
                lidar_features = [lidar_voxel_features, lidar_f_cluster, lidar_f_center]
            else:
                lidar_features = [lidar_voxel_features[..., 3:], lidar_f_cluster, lidar_f_center]
            if self.use_absolute_xyz:
                radar_features = [radar_voxel_features[..., :], radar_f_cluster, radar_f_center] # 这里给去掉了
            else:
                radar_features = [radar_voxel_features[..., 3:], radar_f_cluster, radar_f_center]


            if self.with_distance:
                lidar_points_dist = torch.norm(lidar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                lidar_features.append(lidar_points_dist)
            lidar_features = torch.cat(lidar_features, dim=-1)
            if self.with_distance:
                radar_points_dist = torch.norm(radar_voxel_features[:, :, :3], 2, 2, keepdim=True)
                radar_features.append(radar_points_dist)
            radar_features = torch.cat(radar_features, dim=-1)

            lidar_voxel_count = lidar_features.shape[1]
            radar_voxel_count = radar_features.shape[1]
            lidar_mask = self.get_paddings_indicator(lidar_voxel_num_points, lidar_voxel_count, axis=0)
            radar_mask = self.get_paddings_indicator(radar_voxel_num_points, radar_voxel_count, axis=0)
            lidar_mask = torch.unsqueeze(lidar_mask, -1).type_as(lidar_voxel_features)
            radar_mask = torch.unsqueeze(radar_mask, -1).type_as(radar_voxel_features)
            lidar_features *= lidar_mask
            radar_features *= radar_mask

            # for pfn in self.pfn_layers:
            #     lidar_features = pfn(lidar_features)
            # lidar_features = lidar_features.squeeze()
            # for pfn in self.pfn_layers:
            #     radar_features = pfn(radar_features)
            # radar_features = radar_features.squeeze()
            # 这里速度其他变量给去掉了
            #   ['x', 'y', 'z', 'rcs', 'v_r', 'v_r_comp', 'time']  7个
            
            # safusionlayer2
            #  对radar加上一个特征提取
            add_features_to_map = radar_features[:, :, 3:7]
            
            # add_features_to_map = torch.max(add_features_to_map, dim=1, keepdim=True)[0]
            # add_features_to_map = add_features_to_map.squeeze()
            # batch_dict['add_features_to_map'] = add_features_to_map
            
            # radar_features = self.encoder_radar[0](radar_features)   # 后面是10
            radar_features = radar_features[:,:,[0,1,2,6,7,8,9,10,11,12]]  # 这里感觉还是加上了时间的维度
            
            add_features_to_map = self.encoder_radar[0](add_features_to_map)   # 得到64维度的额外编码 # B,N,C  
            
            add_features_to_map = self.score_fn(add_features_to_map).permute(0,2,1).contiguous()  # B,C,N
            
            
            
            
            lidar_features_output = self.interral(lidar_features, radar_features)
            radar_features_output = self.interral(radar_features, lidar_features) # B,C，N
            
            # 再这里额外对速度做了一个处理，类似于Randanet中软注意力机制
            
            radar_features_output = add_features_to_map*radar_features_output
            radar_features_output = torch.max(radar_features_output, dim=2, keepdim=True)[0]
            
            # 用速度其他属性对lidar进行加权
            lidar_features_output = self.interral_lidar(lidar_features_output.permute(0,2,1).contiguous() , add_features_to_map.permute(0,2,1).contiguous())
            lidar_features_output = torch.max(lidar_features_output, dim=2, keepdim=True)[0]
            
            
            
            lidar_features = lidar_features_output.view([lidar_features_output.size()[0], lidar_features_output.size()[1]])
            radar_features = radar_features_output.view([radar_features_output.size()[0], radar_features_output.size()[1]])

            batch_dict['lidar_pillar_features'] = lidar_features
            batch_dict['radar_pillar_features'] = radar_features
        
        return batch_dict

# class PointNet(nn.Module):
#     def __init__(self,
#                  in_channels,
#                  out_channels,
#                  use_norm=True,
#                  last_layer=False):
#         """
#         in_channels: 10
#         out_channels: 64
#         """
#         super().__init__()
        
#         self.last_vfe = last_layer # True
#         self.use_norm = use_norm # True
#         if not self.last_vfe:
#             out_channels = out_channels // 2

#         if self.use_norm:
#             self.linear = nn.Linear(in_channels, out_channels, bias=False) # 线性层 + BatchNorm
#             self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
#         else:
#             self.linear = nn.Linear(in_channels, out_channels, bias=True)

#         self.part = 50000

#     def forward(self, inputs):
#         """
#         inputs:（31530，32，10)
#         """
#         if inputs.shape[0] > self.part:
#             # nn.Linear performs randomly when batch size is too large
#             num_parts = inputs.shape[0] // self.part
#             part_linear_out = [self.linear(inputs[num_part*self.part:(num_part+1)*self.part])
#                                for num_part in range(num_parts+1)]
#             x = torch.cat(part_linear_out, dim=0)
#         else:
#             # 线性层
#             x = self.linear(inputs) # (31530,32,64)
#         torch.backends.cudnn.enabled = False
#         # BatchNorm1d层:(31530, 64, 32) --> (31530, 32, 64)
#         # 这里之所以变换维度，是因为BatchNorm1d在通道维度上进行,对于图像来说默认模式为[N,C,H*W],通道在第二个维度上
#         x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1) if self.use_norm else x
#         torch.backends.cudnn.enabled = True
#         # 激活函数
#         x = F.relu(x)
#         # 按照维度取每个voxel中的最大值 --> (31530, 1, 64)
#         # 这里的0是表示取数值，max的1表示索引
#         x_max = torch.max(x, dim=1, keepdim=True)[0]

#         if self.last_vfe:
#             return x_max
#         else:
#             # torch的repeat在第几维度复制几遍
#             x_repeat = x_max.repeat(1, inputs.shape[1], 1)
#             # 在最后一个维度上拼接 
#             x_concatenated = torch.cat([x, x_repeat], dim=2)
#             return x_concatenated