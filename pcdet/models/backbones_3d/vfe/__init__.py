from .mean_vfe import MeanVFE
from .pillar_vfe import PillarVFE,PillarVFE_Lidar,PillarVFE_velocity,PillarVFE_randanet,PillarVFE_randanet_radar_lidar,PillarVFE_randanet_radar_lidar_our_attention,PillarVFE_randanet_new,PillarVFE_randanet_new_radar_lidar
from .vfe_template import VFETemplate

__all__ = {
    'VFETemplate': VFETemplate,
    'MeanVFE': MeanVFE,
    'PillarVFE': PillarVFE,
    'PillarVFE_Lidar':PillarVFE_Lidar,
    'PillarVFE_velocity':PillarVFE_velocity,
    'PillarVFE_randanet': PillarVFE_randanet,
    'PillarVFE_randanet_new': PillarVFE_randanet_new,
    'PillarVFE_randanet_new_radar_lidar': PillarVFE_randanet_new_radar_lidar,
    'PillarVFE_randanet_radar_lidar': PillarVFE_randanet_radar_lidar,
    'PillarVFE_randanet_radar_lidar_our_attention': PillarVFE_randanet_radar_lidar_our_attention
}
