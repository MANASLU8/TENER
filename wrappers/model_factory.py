from utils.data_loaders import get_data_loading_function
from models.TENER import TENER

def make_model(model_id, dataset, config):
    if model_id == config['model_ids']['tener']:
        data_bundle, embed = get_data_loading_function(dataset, config)(dataset, config)
        return TENER(data_bundle=data_bundle, config=config, embed=embed, num_layers=config['num_layers'],
                       d_model=config['n_heads'] * config['head_dims'], n_head=config['n_heads'],
                       feedforward_dim=int(2 * config['n_heads'] * config['head_dims']), dropout=config['dropout'],
                        after_norm=config['after_norm'], attn_type=config['attn_type'],
                       bi_embed=None,
                        fc_dropout=config['fc_dropout'],
                       pos_embed=config['pos_embed'],
                      scale=config['attn_type']=='transformer')
