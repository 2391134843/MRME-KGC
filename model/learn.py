import os
import json
import argparse
import numpy as np

import torch
from torch import optim

from datasets import Dataset
from models import *
from regularizers import *
from optimizers import KBCOptimizer
# import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

datasets = ['WN18RR', 'FB237', 'YAGO3-10','CN-100K','CODE-L','CODE-S','Atomic','DB100K',"kinships","UML"]

parser = argparse.ArgumentParser(
    description="Knowledge Graph Completion"
)

parser.add_argument(
    '--dataset', choices=datasets,
    help="Dataset in {}".format(datasets)
)

parser.add_argument(
    '--model', type=str, default='MRME-KGC'
)

parser.add_argument(
    '--regularizer', type=str, default='NA',
)

optimizers = ['Adagrad', 'Adam', 'SGD','DAdaptAdaGrad']
parser.add_argument(
    '--optimizer', choices=optimizers, default='Adagrad',
    help="Optimizer in {}".format(optimizers)
)

parser.add_argument(
    '--max_epochs', default=50, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid', default=3, type=float,
    help="Number of epochs before valid."
)
parser.add_argument(
    '--rank', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--reg', default=0, type=float,
    help="Regularization weight"
)
parser.add_argument(
    '--init', default=1e-3, type=float,
    help="Initial scale"
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--decay1', default=0.9, type=float,
    help="decay rate for the first moment estimate in Adam"
)
parser.add_argument(
    '--decay2', default=0.999, type=float,
    help="decay rate for second moment estimate in Adam"
)

parser.add_argument('-train', '--do_train', action='store_true')
parser.add_argument('-test', '--do_test', action='store_true')
parser.add_argument('-save', '--do_save', action='store_true')
parser.add_argument('-weight', '--do_ce_weight', action='store_true')
parser.add_argument('-path', '--save_path', type=str, default='../logs/')
parser.add_argument('-id', '--model_id', type=str, default='0')
parser.add_argument('-ckpt', '--checkpoint', type=str, default='')

args = parser.parse_args()

if args.do_save:
    assert args.save_path
    save_suffix = args.model + '_' + args.regularizer + '_' + args.dataset + '_' + args.model_id

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    save_path = os.path.join(args.save_path, save_suffix)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

data_path = "../data"
dataset = Dataset(data_path, args.dataset)
examples = torch.from_numpy(dataset.get_train().astype('int64'))

if args.do_ce_weight:
    ce_weight = torch.Tensor(dataset.get_weight()).cuda()
else:
    ce_weight = None

print(dataset.get_shape())

model = None
regularizer = None

exec('model = '+args.model+'(dataset.get_shape(), args.rank, args.init)')

exec('regularizer = '+args.regularizer+'(args.reg)')
regularizer = [regularizer, N3(args.reg)]

device = 'cuda'
model.to(device)
for reg in regularizer:
    reg.to(device)

optim_method = {
    'Adagrad': lambda: optim.Adagrad(model.parameters(), lr=args.learning_rate),
    'Adam': lambda: optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.decay1, args.decay2)),
    'SGD': lambda: optim.SGD(model.parameters(), lr=args.learning_rate)
}[args.optimizer]()

optimizer = KBCOptimizer(model, regularizer, optim_method, args.batch_size)


def avg_both(mrrs: Dict[str, float],mrs:Dict[str,float], hits: Dict[str, torch.FloatTensor]):
    mrr = (mrrs['lhs'] + mrrs['rhs']) / 2.
    mrr_str= "{:.4f}".format(mrr)  
    mr = (mrs['lhs'] + mrs['rhs']) / 2.
    mr_str = "{:.0f}".format(mr)  
    h = (hits['lhs'] + hits['rhs']) / 2.
    # h_lhs = hits['lhs']
    # h_rhs = hits['rhs']
    # return {'MRR': mrr_str,'MR':mr_str, 'hits_h_lhs@[1,3,10]': h_lhs,'hits_h_rhs@[1,3,10]':h_rhs,'lhs_MRR':mrrs['lhs'],'rhs_MRR':mrrs['rhs']}
    return {'MRR': mrr_str,'MR':mr_str, 'hits@[1,3,10]': h,'lhs_MRR':mrrs['lhs'],'rhs_MRR':mrrs['rhs']}



cur_loss = 0
device_ids = [0, 1]
if args.checkpoint is not '':
    model.load_state_dict(torch.load(os.path.join(args.checkpoint, 'checkpoint'), map_location='cuda:0'))
    model = nn.DataParallel(model, device_ids=device_ids)
losses = []
train_mrrs = []
train_mrs= []
train_hit1s = []
train_hit3s = []
train_hit10s= []

vaild_mrrs = []
vaild_mrs= []
vaild_hit1s = []
vaild_hit3s = []
vaild_hit10s= []
if args.do_train:
    with open(os.path.join(save_path, 'train.log'), 'w') as log_file:
        for e in range(args.max_epochs):
            print("Epoch: {}".format(e+1))

            cur_loss = optimizer.epoch(examples, e=e, weight=ce_weight)
            losses.append(cur_loss)

            if (e + 1) % args.valid == 0:
                valid, test, train = [
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]

                print("\t TRAIN: ", train)
                values_list = list(train.values())
                tran_mrr = values_list[0]
                tran_mr = values_list[1]
                tran_hits = values_list[2]
                tran_hit1 = tran_hits[0].item()
                tran_hit3 = tran_hits[1].item()
                tran_hit10 = tran_hits[2].item()
                train_mrrs.append(tran_mrr)
                train_mrs.append(tran_mr)
                train_hit1s.append(tran_hit1)
                train_hit3s.append(tran_hit3)
                train_hit10s.append(tran_hit10)

                # print("train_mrr: ", tran_mrr,"train_mr: ", tran_mr, "train_hit1: ", tran_hit1, "train_hit3: ", tran_hit3, "train_hit10: ", tran_hit10)
                print("\t VALID: ", valid)
                # values_list2 = list(valid.values())
                # vaild_mrr =values_list2[0]
                # vaild_mr = values_list2[1]
                # vaild_hits = values_list2[2]
                # vaild_hit1 = vaild_hits[0].item()
                # vaild_hit3 = vaild_hits[1].item()
                # vaild_hit10 = vaild_hits[2].item()
                # vaild_mrrs.append(vaild_mrr)
                # vaild_mrs.append(vaild_mr)
                # vaild_hit1s.append(vaild_hit1)
                # vaild_hit3s.append(vaild_hit3)
                # vaild_hit10s.append(vaild_hit10)

                # print("vaild_mrr: ", vaild_mrr, "vaild_mr: ", vaild_mr, "vaild_hit1: ", vaild_hit1, "vaild_hit3: ", vaild_hit3, "vaild_hit10: ", vaild_hit10)

                # log_file.write("Epoch: {}\n".format(e+1))
                # log_file.write("\t TRAIN: {}\n".format(train))
                # log_file.write("\t VALID: {}\n".format(valid))
                # log_file.flush()

        test = avg_both(*dataset.eval(model, 'test', 50000))

        print("\t TEST : ", test)
# # 
# cpu_tensor = torch.Tensor(losses).cpu()
# # 
# numpy_array = cpu_tensor.numpy()
# print(numpy_array)
# plt.plot(numpy_array)
# # print("train_mrrs: ", train_mrrs)
# # print("train_mrs: ", train_mrs)
# # print("train_hit1: ", train_hit1s)
# # print("train_hit3: ", train_hit3s)
# # print("train_hits10: ", train_hit10s)
# # print("vaild_mrrs: ", vaild_mrrs)
# # print("vaild_mrs: ", vaild_mrs)
# # print("vaild_hit1: ", vaild_hit1s)
# # print("vaild_hit3: ", vaild_hit3s)
# # print("vaild_hits10: ", vaild_hit10s)
#
# # 
# plt.title('Loss Function Curve')
# plt.xlabel('Training Steps')
# plt.ylabel('Loss')
#
# # 
# plt.show()
# if args.do_save:
#     torch.save(model.state_dict(), os.path.join(save_path, 'checkpoint'))
#     embeddings = model.embeddings
#     len_emb = len(embeddings)
#     if len_emb == 2:
#         np.save(os.path.join(save_path, 'entity_embedding.npy'), embeddings[0].weight.detach().cpu().numpy())
#         np.save(os.path.join(save_path, 'relation_embedding.npy'), embeddings[1].weight.detach().cpu().numpy())
#     elif len_emb == 3:
#         np.save(os.path.join(save_path, 'head_entity_embedding.npy'), embeddings[0].weight.detach().cpu().numpy())
#         np.save(os.path.join(save_path, 'relation_embedding.npy'), embeddings[1].weight.detach().cpu().numpy())
#         np.save(os.path.join(save_path, 'tail_entity_embedding.npy'), embeddings[2].weight.detach().cpu().numpy())
#     else:
#         print('SAVE ERROR!')
#
