import copy
import os
import time
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
from Attempt.train_factory.utils import *
from Attempt.model.rec_model import NPRec
from data_factory.interdata import InterData
from collections import Counter


class ValidatePerformance:
    def __init__(self, data_name, model_config, train_config, test_config):
        self.data_name = data_name
        self.model_name = model_config['model_name']
        self.dataset = InterData(data_name).load_data()
        self.model_config = model_config
        self.train_config = train_config
        self.test_config = test_config
        self.max_len = self.train_config['max_len'][self.data_name]
        self.device = self.train_config['device']
        self.best_metric_dict = {}
        for k in [5, 10]:
            self.best_metric_dict['HR@{}'.format(k)] = 0.0
            self.best_metric_dict['NDCG@{}'.format(k)] = 0.0
        self.best_model = None
        self.bad_count = 0
        self.early_stop_flag = False
        self.result_path_prefix = '../results/'
        self.model_path_prefix = '../models/'

    def construct_train_loader(self):
        train_data = get_train_set(self.dataset, self.max_len)
        batch_size = self.train_config['batch_size'][self.data_name]
        train_loader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  sampler=LadderSampler(train_data, batch_size),
                                  num_workers=12,
                                  prefetch_factor=2,
                                  collate_fn=lambda x: gen_train_batch(x, train_data, self.max_len))
        return train_loader

    def construct_test_loader(self):
        test_data = get_test_set(self.dataset, self.max_len)
        batch_size = self.test_config['batch_size'][self.data_name]
        test_loader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 num_workers=12,
                                 prefetch_factor=2,
                                 collate_fn=lambda x: gen_eval_batch(x, test_data, self.max_len))
        return test_loader

    def construct_model(self):
        model = NPRec(vocab_size=self.dataset.n_item, embed_dim=self.model_config['embed_dim'], num_latent=self.model_config['num_latent'])
        return model

    def make_directions(self):
        result_path = self.result_path_prefix + self.data_name + '/'
        model_path = self.model_path_prefix + self.data_name + '/'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return result_path, model_path

    def train_model(self, continue_train=False):
        fix_random_seed_as(self.train_config['random_seed'])
        train_loader = self.construct_train_loader()
        test_loader = self.construct_test_loader()
        model = self.construct_model()
        result_path, model_path = self.make_directions()
        result_save_path = result_path + self.model_name + '.txt'
        model_save_path = model_path + self.model_name + '.pkl'
        optimizer = torch.optim.Adam(model.parameters(), lr=self.train_config['learning_rate'])
        if continue_train:
            model.load(model_save_path)
        model.to(self.device)
        for i in range(1, self.train_config['epochs']):
            self.train_epoch(i, train_loader, model, optimizer)
            if i > self.train_config['test_interval'] and (i % self.train_config['test_interval']) == 0:
                current_metric = self.calculate_metrics(i, test_loader, model)
                self.early_stop_flag = self.indicator(current_metric, model, result_save_path, model_save_path)
                if self.early_stop_flag:
                    print('Early Stopped at Epoch {} !'.format(i))
                    break

    def train_epoch(self, epoch, loader, model, optimizer):
        print('>' * 30, 'Training Epoch: {} ...'.format(epoch))
        start_time = time.time()
        model.train()
        running_loss = 0.0
        processed_batch = 0
        batch_iterator = tqdm(enumerate(loader), total=len(loader), leave=True)
        for batch_idx, (src_seq, trg_seq, data_size) in batch_iterator:
            optimizer.zero_grad()
            src = src_seq.to(self.device)
            trg = trg_seq.to(self.device)
            dsz = data_size.to(self.device)
            logits, kl = model.train_model(src, trg)
            b, n, t, N = logits.size()
            logits = logits.view(-1, logits.size(-1))
            target = trg.view(-1)
            target = trg.unsqueeze(-1).expand(-1, -1, t)
            target = target.reshape(-1)
            loss = F.cross_entropy(logits, target, ignore_index=0) + 0.01 * kl
            loss.backward()
            optimizer.step()
            running_loss += loss.detach().cpu().item()
            processed_batch = processed_batch + 1
            batch_iterator.set_postfix_str('Loss={:.4f}'.format(loss.item()))
        cost_time = time.time() - start_time
        avg_loss = running_loss / processed_batch
        print('Time={:.4f}, Average Loss={:.4f}'.format(cost_time, avg_loss))

    def calculate_metrics(self, epoch, loader, model):
        if epoch is None:
            print('>' * 30, 'Evaluating {} on {} ...'.format(self.model_name, self.data_name))
        else:
            print('>' * 30, 'Evaluating Epoch: {} ...'.format(epoch))
        cnt = Counter()
        array = np.zeros(self.dataset.n_item)
        model.eval()
        batch_iterator = tqdm(enumerate(loader), total=len(loader), leave=True)
        for batch_idx, (src_seq, ctx_seq, trg_seq, data_size) in batch_iterator:
            src = src_seq.to(self.device)
            ctx = ctx_seq.to(self.device)
            trg = trg_seq.to(self.device)
            dsz = data_size.to(self.device)
            logits = model.inference(src, ctx, dsz)
            cnt = count(logits, trg, cnt)
        hr, ndcg = calculate(cnt, array)
        mertic = dict()
        for k in [5, 10]:
            mertic['HR@{}'.format(k)] = hr[k - 1]
            mertic['NDCG@{}'.format(k)] = ndcg[k - 1]
        return mertic

    def indicator(self, current_metric, current_model, result_save_path, model_save_path):
        metric_indicator = 0
        for metric in current_metric.keys():
            if current_metric[metric] > self.best_metric_dict[metric]:
                self.best_model = copy.deepcopy(current_model)
                self.best_metric_dict[metric] = current_metric[metric]
                print('Imp ! Current {} = {:.2f} % v.s. Best {} = {:.2f} %'.format(metric, current_metric[metric] * 100,
                                                                             metric, self.best_metric_dict[metric] * 100))
            else:
                metric_indicator += 1
                print('Dec ! Current {} = {:.2f} % v.s. Best {} = {:.2f} %'.format(metric, current_metric[metric] * 100,
                                                                            metric, self.best_metric_dict[metric] * 100))

        if metric_indicator == 4:
            self.bad_count += 1
        else:
            self.bad_count = 0
            self.best_model.save(model_save_path)
            f = open(result_save_path, 'w')
            for metric, value in self.best_metric_dict.items():
                f.write('Best {} = {:.2f} %'.format(metric, self.best_metric_dict[metric] * 100))
                f.write('\n')
            f.close()
        if self.bad_count > self.train_config['patience']:
            early_stop_flag = True
        else:
            early_stop_flag = False
        print('Bad Count = {}'.format(self.bad_count))
        return early_stop_flag

    def forward_only(self):
        model = self.construct_model()
        model_name = self.model_name
        path = self.model_path_prefix + self.data_name + '/' + model_name + '.pkl'
        model.load(path)
        model.to(self.device)
        test_loader = self.construct_test_loader()
        metric = self.calculate_metrics(None, test_loader, model)
        result_path = self.result_path_prefix + self.data_name + '/' + model_name + '.txt'
        f = open(result_path, 'w')
        print("Dataset : {}".format(self.data_name), file=f)
        print("Model : {}".format(model_name), file=f)
        for k, v in metric.items():
            print('{} = {:.2f} %'.format(k, v * 100))
            print('{} = {:.2f} %'.format(k, v * 100), file=f)
        f.close()


if __name__ == '__main__':
    model_config = {'model_name': None,
                    'embed_dim': 64,
                    'num_latent': 10,}
    train_config = {'batch_size': {'ml1m': 64},
                    'max_len': {'ml1m': 160},
                    'device': 'cuda:0',
                    'learning_rate': 0.001,
                    'test_interval': 1,
                    'patience': 50,
                    'random_seed': 42,
                    'epochs': 1000}
    test_config = {'batch_size': {'ml1m': 128}}

    latent_list = [1, 5, 10, 15]
    for data_name in ['nyc']:
        for num_latent in latent_list:
            model_name = 'NPRec_wo_seq_' + str(num_latent)
            model_config['model_name'] = model_name
            model_config['num_latent'] = num_latent
            trainer = ValidatePerformance(data_name, model_config, train_config, test_config)
            trainer.train_model(False)
            trainer.forward_only()
