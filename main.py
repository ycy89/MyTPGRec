import torch
from trainer import Trainer
import dataloader
from model import Model
import argparse
import os
import random
import numpy as np
from logger import init_logger

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='baby', help='baby/clothing/sports/beauty')
parser.add_argument('--train_batch_size', type=int, default=2048)
parser.add_argument('--test_batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--data_dir', type=str, default='./data')
parser.add_argument('--save_dir', type=str, default='./save_model')
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--layers', type=int, default=2, help='GNN layers')
parser.add_argument('--reg_weight', type=float, default=1e-5)
parser.add_argument('--ssl_temp', type=float, default=0.5, help='temperature for CL_loss')
parser.add_argument('--ssl_reg', type=float, default=.03)
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--metrics', type=list, default=['Recall', 'NDCG'])
parser.add_argument('--topk', type=list, default=[10, 20])
parser.add_argument('--edge_drop_rate', type=float, default=0.3)
parser.add_argument('--edge_add_rate', type=float, default=.2)
parser.add_argument('--use_text', action='store_false')
parser.add_argument('--Sem_CL', action='store_false')
parser.add_argument('--Str_CL', action='store_false')

args, _ = parser.parse_known_args()

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    logger = init_logger(args.dataset)
    logger.info(args)
    device = torch.device(args.device)
    same_seeds(2024)
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, train_dict, val_data, test_data, text_feat, uvs, ivs, user_nids, item_nids, edge_p = \
        dataloader.get_bpr_loaders(args, n_w=0)
    model = Model(uvs, ivs, text_feat, edge_p, user_nids, item_nids, args).to(device)

    ## training
    trainer = Trainer(args, model, device, logger)
    logger.info('training...')
    trainer.training(train_loader, train_dict, val_data, test_data, test_interval=1)
    logger.info('train done')