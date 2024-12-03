import array
import gzip
import json
import os
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer


np.random.seed(123)

# folder = './CD2014/'
# name = 'CDs_and_Vinyl'
# folder = './movie/'
# name = 'Movies_and_TV'
# folder = './Electronics/'
# name = 'Electronics'
# folder = './kindle/'
# name = 'Kindle_Store'
folder = './toys/'
name = 'Toys_and_Games'
# folder = './beauty/'
# name = 'Beauty'
# folder = './book2018/'
# name = 'Books'
bert_path = './sentence-bert/all-mpnet-base-v2/'
bert_model = SentenceTransformer(bert_path)
core = 5

def parse(path):
    g = gzip.open(path, 'r')
    for l in g:
        yield json.dumps(eval(l))

print("----------parse metadata----------")
if not os.path.exists(folder + "meta-data/meta.json"):
    with open(folder + "meta-data/meta.json", 'w') as f:
        for l in parse(folder + 'meta-data/' + "meta_%s.json.gz"%(name)):
            f.write(l+'\n')

print("----------parse data----------")
if not os.path.exists(folder + "meta-data/%d-core.json" % core):
    with open(folder + "meta-data/%d-core.json" % core, 'w') as f:
        for l in parse(folder + 'meta-data/' + "reviews_%s_%d.json.gz"%(name, core)):
            f.write(l+'\n')

print("----------load data----------")
jsons = []

for line in open(folder + "meta-data/%d-core.json" % core).readlines():
    jsons.append(json.loads(line))

print("----------Build dict----------")
items = set()
users = set()
for j in jsons:
    items.add(j['asin'])
    users.add(j['reviewerID'])
print("n_items:", len(items), "n_users:", len(users))


item2id = {}
with open(folder + 'item_list.txt', 'w') as f:
    for i, item in enumerate(items):
        item2id[item] = i
        f.writelines(item+'\t'+str(i)+'\n')

user2id =  {}
with open(folder + 'user_list.txt', 'w') as f:
    for i, user in enumerate(users):
        user2id[user] = i
        f.writelines(user+'\t'+str(i)+'\n')


ui = defaultdict(list)
for j in jsons:
    u_id = user2id[j['reviewerID']]
    i_id = item2id[j['asin']]
    ui[u_id].append(i_id)
with open(folder + 'user-item-dict.json', 'w') as f:
    f.write(json.dumps(ui))


print("----------Split Data----------")
train_json = {}
val_json = {}
test_json = {}
for u, items in ui.items():
    if len(items) < 10:
        testval = np.random.choice(len(items), 2, replace=False)
    else:
        testval = np.random.choice(len(items), int(len(items) * 0.2), replace=False)

    test = testval[:len(testval)//2]
    val = testval[len(testval)//2:]
    train = [i for i in list(range(len(items))) if i not in testval]
    train_json[u] = [items[idx] for idx in train]
    val_json[u] = [items[idx] for idx in val.tolist()]
    test_json[u] = [items[idx] for idx in test.tolist()]

with open(folder + 'train.json', 'w') as f:
    json.dump(train_json, f)
with open(folder + 'val.json', 'w') as f:
    json.dump(val_json, f)
with open(folder + 'test.json', 'w') as f:
    json.dump(test_json, f)

print("----------Text Features----------")
raw_text = {}
with open(folder + "meta-data/meta.json", 'r') as f:
    for line in f.readlines():
        meta = json.loads(line)
        if meta['asin'] in item2id:
            string = ' '
            if 'categories' in meta:    #  category->2018    categories->2014
                for cates in meta['categories']:
                    for cate in cates:
                        string += cate + ' '
            if 'title' in meta:
                string += meta['title']
            if 'brand' in meta:
                string += meta['brand']
            if 'description' in meta:            #  2018->[0]
                # for i in meta['description']:
                string += meta['description']
            raw_text[item2id[meta['asin']]] = string.replace('\n', ' ')
texts = []
for i in range(len(item2id)):
    texts.append(raw_text[i] + '\n')
bs, l = 512, 0
sentence_embeddings = None
while l < len(texts):
    r = l + bs if l + bs <= len(texts) else len(texts)
    embs = bert_model.encode(texts[l:r])
    sentence_embeddings = np.concatenate((sentence_embeddings, embs), axis=0) if sentence_embeddings is not None else embs
    l += bs
    print(l, ' items done')
sentence_embeddings = bert_model.encode(texts)
assert sentence_embeddings.shape[0] == len(item2id)
np.save(folder+'item_text_feat.npy', sentence_embeddings)
