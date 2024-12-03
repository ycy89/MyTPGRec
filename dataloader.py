import json
import os
import numpy as np
import torch
import torch.utils.data as Data
import random

class BPRDataset(Data.Dataset):
    def __init__(self, train_dict, dir, uvs, ivs, user_nids, item_nids, u_degree, i_degree, add_rate):
        super().__init__()
        self.uvs = uvs
        self.ivs = ivs
        self.data_dir = dir
        self.train_dict = train_dict
        self.u_degree, self.i_degree = u_degree, i_degree
        self.user_nids, self.item_nids = torch.tensor(user_nids), torch.tensor(item_nids)
        self.counter = 0
        np_array = np.array([i_degree[i] for i in range(ivs)])
        self.sorted_indices = np.argsort(np_array)

        self.edge_p = torch.tensor([(self.u_degree[user_nids[i]] ** -5e-1) *
                                    (self.i_degree[item_nids[i]] ** -5e-1) for i in range(len(user_nids))]) # edge_p for adding cold edges

        self.add_rate = add_rate
        self.all_len = len(self.user_nids) + int(self.add_rate * len(self.user_nids))  # adding extra training edges

    def __getitem__(self, idx):
        # self.counter & self.idx are used for adding extra training edges
        if not hasattr(self, 'idx') or self.counter == self.__len__():
            self.idx = torch.multinomial(self.edge_p, int(self.add_rate * len(self.user_nids)), replacement=True)
            self.counter = 0
        self.counter += 1
        if idx < len(self.user_nids):
            u, i = self.user_nids[idx], self.item_nids[idx]
        else:
            u, i = self.user_nids[self.idx[idx - len(self.user_nids)]], self.item_nids[self.idx[idx - len(self.user_nids)]]

        part = 0.8
        neg = self.sorted_indices[random.randint(int(self.ivs * part), self.ivs - 1)]
        while neg in self.train_dict[int(u)]:
            neg = self.sorted_indices[random.randint(int(self.ivs * part), self.ivs - 1)]

        neg2 = self.sorted_indices[random.randint(0, int(self.ivs * part) - 1)]
        while neg2 in self.train_dict[int(u)]:
            neg2 = self.sorted_indices[random.randint(0, int(self.ivs * part) - 1)]

        return u, i, torch.tensor(neg), torch.tensor(neg2)

    def __len__(self):
        return self.all_len

def get_bpr_loaders(args, n_w):
    file_dir = os.path.join(args.data_dir, args.dataset)
    user_nids, item_nids = [], []
    u_degree, i_degree = {}, {}
    item_set = set()
    with open(os.path.join(file_dir, 'train.json'), 'r') as f:
        train_dict = json.load(f)
        train_dict = {int(key): value for key, value in train_dict.items()}
        uvs = len(train_dict)
        for user, items in train_dict.items():
            user_nids.extend([user] * len(items))
            u_degree[user] = len(items)
            item_nids.extend(items)
            item_set.update(items)
            for item in items:
                i_degree[item] = i_degree[item] + 1 if item in i_degree else 1
        f.close()

    with open(os.path.join(file_dir, 'test.json'), 'r') as f:
        test_dict = json.load(f)
        test_data = [[]] * uvs
        for key, value in test_dict.items():
            test_data[int(key)] = value
            item_set.update(value)
        f.close()

    with open(os.path.join(file_dir, 'val.json'), 'r') as f:
        val_dict = json.load(f)
        val_data = [[]] * uvs
        for key, value in val_dict.items():
            val_data[int(key)] = value
            item_set.update(value)
        f.close()
    ivs = len(item_set)
    for i in range(ivs):
        if i not in i_degree:
            i_degree[i] = 0

    if os.path.exists(os.path.join(file_dir, 'item_text_feat.npy')):
        text_feat = np.load(os.path.join(file_dir, 'item_text_feat.npy'))
    else:
        raise Exception('item_text_feat.npy not exists!')

    dataset = BPRDataset(train_dict, args.data_dir, uvs, ivs, user_nids, item_nids, u_degree, i_degree, args.edge_add_rate)
    train_loader = Data.DataLoader(dataset, args.train_batch_size, shuffle=True, num_workers=n_w)

    return train_loader, train_dict, val_data, test_data, text_feat, uvs, ivs, user_nids, item_nids, dataset.edge_p


if __name__ == '__main__':
    ### test code
    pass