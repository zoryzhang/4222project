import LightGCN.code.world as world
import LightGCN.code.utils as utils
from LightGCN.code.world import cprint
import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import time
import LightGCN.code.Procedure as Procedure
import os
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import LightGCN.code.register as register
from LightGCN.code.register import dataset

#os.environ['CUDA_VISIBLE_DEVICES'] ='0'

#STACKING_FUNC = 0
def sanity_check(stacking_func=0, n_layer=3):
    world.config['stacking_func'] = stacking_func
    world.config['lightGCN_n_layers'] = n_layer
    sf = world.config['stacking_func']
    cprint(f'stacking_func: {sf}')
    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    return Recmodel

def run_lightgcn(stacking_func, n_layer=3):
    world.config['stacking_func'] = stacking_func
    world.config['lightGCN_n_layers'] = n_layer
    sf = world.config['stacking_func']
    cprint(f'stacking_func: {sf}')

    Recmodel = register.MODELS[world.model_name](world.config, dataset)
    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)
    """
    world.config['alphas'] = nn.Parameter(torch.Tensor(world.config['lightGCN_n_layers']+1, 1)).to('cuda')
    nn.init.xavier_uniform_(world.config['alphas'])
    if world.config['stacking_func']==3:
        nn.init.constant_(world.config['alphas'], 1/(world.config['lightGCN_n_layers']+1))
    """

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}")
        except FileNotFoundError:
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1

    # init tensorboard
    if world.tensorboard:
        print("we have tensorboard!")
        w : SummaryWriter = SummaryWriter(
                                        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")

    try:
        for epoch in range(world.TRAIN_epochs):
            start = time.time()
            if epoch %10 == 0:
                cprint("[TEST]")
                Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            #print(world.config['alphas'])

            #torch.save(Recmodel.state_dict(), weight_file)
    finally:
        if world.tensorboard:
            w.close()