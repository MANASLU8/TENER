from utils.dataset_loaders import read_dataset
from fastNLP.embeddings import StaticEmbedding, StackEmbedding, LSTMCharEmbedding
from fastNLP import cache_results
from fastNLP.embeddings import CNNCharEmbedding 

def get_data_loading_function(dataset, config):
    name = 'caches/{}_{}_{}_{}_{}.pkl'.format(dataset, config['model_type'], config['encoding_type'], config['char_type'], config['normalize_embed'])

    @cache_results(name, _refresh=config['ignore_cache'])
    def load_data(dataset, config):
        # 替换路径
        data = read_dataset(dataset, config)
        char_embed = None
        if config['char_type'] == 'cnn':
            char_embed = CNNCharEmbedding(vocab=data.get_vocab('chars'), embed_size=30, char_emb_size=30, filter_nums=[30],
                                          kernel_sizes=[3], word_dropout=0, dropout=0.3, pool_method='max'
                                          , include_word_start_end=False, min_char_freq=2)
        elif config['char_type'] in ['adatrans', 'naive']:
            char_embed = TransformerCharEmbed(vocab=data.get_vocab('chars'), embed_size=30, char_emb_size=30, word_dropout=0,
                     dropout=0.3, pool_method='max', activation='relu',
                     min_char_freq=2, requires_grad=True, include_word_start_end=False,
                     char_attn_type=config['char_type'], char_n_head=3, char_dim_ffn=60, char_scale=config['char_type']=='naive',
                     char_dropout=0.15, char_after_norm=True)
        elif config['char_type'] == 'lstm':
            char_embed = LSTMCharEmbedding(vocab=data.get_vocab('chars'), embed_size=30, char_emb_size=30, word_dropout=0,
                     dropout=0.3, hidden_size=100, pool_method='max', activation='relu',
                     min_char_freq=2, bidirectional=True, requires_grad=True, include_word_start_end=False)
        word_embed = StaticEmbedding(vocab=data.get_vocab('chars'),
                                     model_dir_or_name='ru' if dataset.split('/')[-1] in config['datasets']['ru'] else 'en-glove-6b-100d',
                                     requires_grad=True, lower=True, word_dropout=0, dropout=0.5,
                                     only_norm_found_vector=config['normalize_embed'])
        if char_embed is not None:
            embed = StackEmbedding([word_embed, char_embed], dropout=0, word_dropout=0.02)
        else:
            word_embed.word_drop = 0.02
            embed = word_embed

        data.rename_field('words', 'chars')
        return data, embed

    return load_data