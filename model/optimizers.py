import tqdm
import torch
from torch import nn
from torch import optim

from models import KBCModel
import torch.nn.functional as F
from regularizers import Regularizer
from Prodigy import Prodigy

class KBCOptimizer(object):
    def __init__(
            self, model: KBCModel, regularizer: list, optimizer: optim.Optimizer, batch_size: int = 256,
            verbose: bool = True
    ):
        self.model = model
        self.regularizer = regularizer[0]
        self.regularizer2 = regularizer[1]
        self.optimizer = optimizer
        # self.optimizer = Prodigy (model.parameters(), lr=0.1, weight_decay=0.0001)
        print(self.optimizer)
        self.batch_size = batch_size
        self.verbose = verbose

    def epoch(self, examples: torch.LongTensor, e=0, weight=None):
        self.model.train()
        actual_examples = examples[torch.randperm(examples.shape[0]), :]
        loss = nn.CrossEntropyLoss(reduction='mean', weight=weight)


        with tqdm.tqdm(total=examples.shape[0], unit='ex', disable=not self.verbose,ncols=-1) as bar:
            bar.set_description(f'train loss')
            b_begin = 0
            
            while b_begin < examples.shape[0]:
                input_batch = actual_examples[
                    b_begin:b_begin + self.batch_size 
                ].cuda()

                predictions1, factors, contrastive_learning_loss= self.model.forward(input_batch) 
                # predictions1, factors= self.model.forward(input_batch) 
                truth = input_batch[:, 2]

                l_fit = loss(predictions1, truth)
                # print(predictions1, truth)
                # print(truth)
                # l_fit = nn.L1Loss(predictions1, truth)

                l_reg = self.regularizer.forward(factors)

                l = l_fit + l_reg+contrastive_learning_loss   
                

                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()  
                b_begin += self.batch_size
                bar.update(input_batch.shape[0])
                # bar = tqdm(total=total_steps, )

                bar.set_postfix(loss=f'{l.item():.2f}', reg=f'{l_reg.item():.2f}') 

                

        return l
