import copy
import metrics
import numpy as np
import torch
import os

class Trainer(object):
    def __init__(self, args, model, device, logger):
        self.epochs = args.epochs
        self.model = model
        self.device = device
        self.lr = args.lr
        self.opt = torch.optim.Adam(model.parameters(), lr=self.lr)

        self.save_path = os.path.join(args.save_dir, args.dataset + '.pt')
        self.best_result = [0 for _ in range(len(args.metrics) * len(args.topk))]
        self.test_batch_size = args.test_batch_size
        self.logger = logger
        self.metrics = args.metrics
        self.topk = args.topk
        self.edge_drop_rate = args.edge_drop_rate
        self.early_stop = args.early_stop

    def training(self, train_loader, train_data, val_data, test_data, test_interval=3):
        stop_cnt = 0
        self.num_head, self.num_tail = 0, 0
        self.tail_limit = train_loader.dataset.sorted_indices
        self.tail_limit = train_loader.dataset.i_degree[self.tail_limit[int(0.8 * train_loader.dataset.ivs)]]
        for gt_items in test_data:
            for item in gt_items:
                if train_loader.dataset.i_degree[item] <= self.tail_limit:
                    self.num_tail += 1
                else:
                    self.num_head += 1
        self.logger.info(f'test_item: head_num:{self.num_head}   tail_num:{self.num_tail}   limit:{self.tail_limit}')

        for epoch in range(self.epochs):
            self.logger.info('-' * 40)
            self.logger.info(f'Epoch {epoch + 1} ')
            self.model.edge_dropout(self.edge_drop_rate)

            # gamma = max(0.8 - (epoch / 80.0), 0.45)
            # gamma = max(0.8 - np.exp(epoch / 80.0) + 1, 0.45)
            gamma = (1 - epoch / self.epochs) * 0.8 + epoch / self.epochs * 0.5
            print(f'gamma:{gamma}')
            train_loss, show_loss = self.train_an_epoch(train_loader, gamma)
            self.logger.info(f'Loss: {train_loss},  Show_loss: {show_loss}')

            if (epoch + 1) % test_interval == 0:
                self.logger.info('val result')
                _ = self.testing(train_data, val_data, train_loader.dataset.i_degree)
                self.logger.info('test result')
                result = self.testing(train_data, test_data, train_loader.dataset.i_degree)
                times = self.save_model(result, epoch + 1)
                stop_cnt = times * stop_cnt + times

            if stop_cnt > self.early_stop:
                break

    def train_an_epoch(self, train_loader, gamma):
        self.model.train()
        total_loss, total_show_loss = 0, 0

        for interaction in train_loader:
            self.opt.zero_grad()
            loss, show_loss = self.model.calculate_loss(interaction, gamma)
            loss.backward()
            self.opt.step()
            total_loss += loss.item()
            total_show_loss += show_loss
        return total_loss / len(train_loader), total_show_loss / len(train_loader)

    def save_model(self, result, epoch):
        if sum([self.best_result[i] < result[i] for i in range(len(result))]) > len(self.topk):
            self.best_result = result
            state = {'model':self.model.state_dict(), 'epoch':epoch}
            torch.save(state, self.save_path)
            self.logger.info('best model saved')
            return 0
        return 1

    @torch.no_grad()
    def testing(self, train_data, test_data, i_degree):
        user_all_embeddings, item_all_embeddings = self.model.predict()
        max_uid = user_all_embeddings.shape[0]
        max_iid = item_all_embeddings.shape[0]
        result = {}
        for metric in self.metrics:
            for k in self.topk:
                result.update({metric + str(k):.0})

        with torch.no_grad():
            l = 0
            while l < max_uid:
                r = max_uid if l + self.test_batch_size > max_uid else l + self.test_batch_size
                test_u = torch.arange(l,r).repeat_interleave(max_iid, dim=0)
                test_i = torch.arange(max_iid).repeat(r-l)
                pred_scores = torch.sum(user_all_embeddings[test_u] * (item_all_embeddings[test_i]), dim=-1)
                pred_scores = pred_scores.view(-1, max_iid)
                for i in range(r - l):
                    pred_scores[i][train_data[l + i]] = -torch.inf
                _, indice = torch.topk(pred_scores, max(self.topk))
                indices = torch.cat([indices, indice], dim=0) if l > 0 else indice
                l += self.test_batch_size

        all_result = map(self.rank_one_user, zip(indices, test_data, [self.topk] * max_uid, [i_degree] * max_uid))
        tail, head = copy.deepcopy(result), copy.deepcopy(result)

        for one_user_result in all_result:
            overall, head_batch, tail_batch = one_user_result
            for key in result.keys():
                result[key] += overall[key] / max_uid
                head[key] += head_batch[key] / self.num_head
                tail[key] += tail_batch[key] / self.num_tail


        for i in self.topk:
            output = ''
            for metric in self.metrics:
                output += f"{metric}@{i}: {result[metric+str(i)]}  "
            self.logger.info(output)
        for i in self.topk:
            self.logger.info(f"Head Recall@{i}: {head['Recall'+str(i)]}, Tail Recall@{i}: {tail['Recall'+str(i)]}")
        # self.logger.info(f"Recall@50: {result['Recall_50']}, NDCG@50: {result['NDCG_50']}")
        return [result[f'{metric}{i}'] for metric in self.metrics for i in self.topk]
        # return NDCG_20

    def rank_one_user(self, x):
        indices, gt_items, topk, degree = x
        result = {}
        result.update({'Recall' + str(i): .0 for i in topk})
        result.update({'NDCG' + str(i): .0 for i in topk})
        tail = copy.deepcopy(result)
        head = copy.deepcopy(result)
        indices = indices.tolist()

        for gt_item in gt_items:
            for k in topk:
                recall, ndcg = metrics.hit(gt_item, indices[:k]), metrics.ndcg(gt_item, indices[:k])
                result['Recall' + str(k)] += recall / len(gt_items)
                result['NDCG' + str(k)] += ndcg
                if (degree[gt_item] <= self.tail_limit):
                    tail['Recall' + str(k)] += recall
                    # tail['NDCG' + str(k)] += ndcg
                else:
                    head['Recall' + str(k)] += recall
                    # head['NDCG' + str(k)] += ndcg

        for k in topk:
            result['NDCG' + str(k)] /= np.sum(1.0 / np.log2(np.arange(min(len(gt_items), k)) + 2))
        return result, head, tail