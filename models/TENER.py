from torch import optim, load
from fastNLP import Trainer, Tester, GradientClipCallback, WarmupCallback
from modules.callbacks import EvaluateCallback
from fastNLP import SpanFPreRecMetric, BucketSampler
import fastNLP

from fastNLP.modules import ConditionalRandomField, allowed_transitions
from modules.transformer import TransformerEncoder

from torch import nn
import torch
import torch.nn.functional as F

from utils.dataset_loaders import read_dataset
from fastNLP.core.predictor import Predictor
from utils.file_operations import write_lines
from utils.formatter import flatten_prediction_results, get_unique_targets
from metrics.average_precision import get_average_precision

class TENER(nn.Module):
    def __init__(self, config, data_bundle, embed, num_layers, d_model, n_head, feedforward_dim, dropout,
                 after_norm=True, attn_type='adatrans',  bi_embed=None,
                 fc_dropout=0.3, pos_embed=None, scale=False, dropout_attn=None):
        """

        :param tag_vocab: fastNLP Vocabulary
        :param embed: fastNLP TokenEmbedding
        :param num_layers: number of self-attention layers
        :param d_model: input size
        :param n_head: number of head
        :param feedforward_dim: the dimension of ffn
        :param dropout: dropout in self-attention
        :param after_norm: normalization place
        :param attn_type: adatrans, naive
        :param rel_pos_embed: position embedding的类型，支持sin, fix, None. relative时可为None
        :param bi_embed: Used in Chinese scenerio
        :param fc_dropout: dropout rate before the fc layer
        """
        super().__init__()
        self.config = config
        self.data_bundle = data_bundle
        tag_vocab = data_bundle.get_vocab('target')
        self.embed = embed
        embed_size = self.embed.embed_size
        self.bi_embed = None
        if bi_embed is not None:
            self.bi_embed = bi_embed
            embed_size += self.bi_embed.embed_size

        self.in_fc = nn.Linear(embed_size, d_model)

        self.transformer = TransformerEncoder(num_layers, d_model, n_head, feedforward_dim, dropout,
                                              after_norm=after_norm, attn_type=attn_type,
                                              scale=scale, dropout_attn=dropout_attn,
                                              pos_embed=pos_embed)
        self.fc_dropout = nn.Dropout(fc_dropout)
        self.out_fc = nn.Linear(d_model, len(tag_vocab))
        trans = allowed_transitions(tag_vocab, include_start_end=True)
        self.crf = ConditionalRandomField(len(tag_vocab), include_start_end_trans=True, allowed_transitions=trans)
        

    def _forward(self, chars, target, bigrams=None):
        mask = chars.ne(0)
        chars = self.embed(chars)
        if self.bi_embed is not None:
            bigrams = self.bi_embed(bigrams)
            chars = torch.cat([chars, bigrams], dim=-1)

        chars = self.in_fc(chars)
        chars = self.transformer(chars, mask)
        chars = self.fc_dropout(chars)
        chars = self.out_fc(chars)
        logits = F.log_softmax(chars, dim=-1)
        if target is None:
            paths, _ = self.crf.viterbi_decode(logits, mask)
            return {'pred': paths}
        else:
            loss = self.crf(logits, target, mask)
            return {'loss': loss}

    def forward(self, chars, target, bigrams=None):
        return self._forward(chars, target, bigrams)

    def predict(self, chars, bigrams=None):
        return self._forward(chars, target=None, bigrams=bigrams)

    def _get_trainer(self, models_folder):
        optimizer = optim.SGD(self.parameters(), lr=self.config['lr'], momentum=0.9)

        callbacks = []
        clip_callback = GradientClipCallback(clip_type='value', clip_value=5)
        evaluate_callback = EvaluateCallback(self.data_bundle.get_dataset('test'))

        if self.config['warmup_steps']>0:
            warmup_callback = WarmupCallback(self.config['warmup_steps'], schedule='linear')
            callbacks.append(warmup_callback)
        callbacks.extend([clip_callback, evaluate_callback])

        return Trainer(self.data_bundle.get_dataset('train'), self, optimizer, batch_size=self.config['batch_size'], sampler=BucketSampler(),
                          num_workers=2, n_epochs=100, dev_data=self.data_bundle.get_dataset('dev'),
                          metrics=SpanFPreRecMetric(tag_vocab=self.data_bundle.get_vocab('target'), encoding_type=self.config['encoding_type']),
                          dev_batch_size=self.config['batch_size']*5, callbacks=callbacks, device=self.config['device'], test_use_tqdm=False,
                          use_tqdm=True, print_every=300, save_path=models_folder)

    def train_model(self, models_folder):
        trainer = self._get_trainer(models_folder)
        trainer.train(load_best_model=False)

    def load(self, path):
        trainer = self._get_trainer('/'.join(path.split('/')[:-1]))

        load_succeed = trainer._load_model(self, path.split('/')[-1])

        if load_succeed:
            print("Reloaded trained model.")
        else:
            print("Fail to reload trained model.")

        return self

    def test(self, dataset, subset):
        metrics_to_test = [fastNLP.core.metrics.AccuracyMetric()]

        # Load dataset for testing
        databundle_for_test = read_dataset(dataset, self.config)

        # Perform testing
        tester = Tester(databundle_for_test.get_dataset(subset), self, metrics_to_test, batch_size=self.config['batch_size'], num_workers=0, device=None, verbose=1, use_tqdm=True)
        tester.test()

        flattened_true_entities, flattened_predicted_entities = flatten_prediction_results(
            self.data_bundle,
            databundle_for_test,
            subset,
            self._predict(subset_for_prediction = databundle_for_test.get_dataset(subset), targets = self.data_bundle.vocabs["target"], filename = None)
        )

        print("Precision per label:")
        labels = get_unique_targets(self.data_bundle.vocabs["target"])
        scores = get_average_precision(y_true=flattened_true_entities, y_pred=flattened_predicted_entities, labels=labels, average=None)
        for label, score in zip(labels, scores):
            print(f'{label:10s} {score:.2f}')
        
        #print(get_average_precision(flattened_true_entities, flattened_predicted_entities, 'weighted'))
        #for averaging_method in ['micro', 'macro', 'weighted', 'samples']:
            #print(averaging_method)
            #print(get_average_precision(flattened_true_entities, flattened_predicted_entities, averaging_method))

        # print(len(flattened_predicted_entities))
        # print(len(flattened_true_entities))



    def _predict(self, subset_for_prediction, targets, filename):
        predictor = Predictor(self)
        predictions = predictor.predict(subset_for_prediction)['pred']
        words = list(subset_for_prediction.get_field('raw_words'))
        lines = []

        words_sequence_index = 1
        labels_sequence_index = 0
        for sentence in list(zip(predictions, words)):
          if type(sentence[labels_sequence_index][0]) == int:
            continue
          words = sentence[words_sequence_index]
          #print(sentence[labels_sequence_index])
          labels = map(lambda label: f'{targets.to_word(label).split("-")[-1]}', sentence[labels_sequence_index][0])
          for pair in zip(words, labels):
            lines.append('\t'.join(pair))
          lines.append('')
        if filename is not None:
            write_lines(filename, lines)
        return lines

    def export_predictions(self, dataset, subset, output_file):
        # Load dataset for prediction
        databundle_for_prediction = read_dataset(dataset, self.config)

        # Perform prediction
        return self._predict(databundle_for_prediction.get_dataset(subset), self.data_bundle.vocabs["target"], output_file)