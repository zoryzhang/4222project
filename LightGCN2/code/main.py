import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
from ckpt import State
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

# ====================== our changes ======================
weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
state = State(bpr.model, bpr.opt, weight_file)
# ====================== our changes ======================

if world.LOAD:
    try:
        state.load(world.device)
        # Recmodel.load_state_dict(torch.load(weight_file, map_location=world.device))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    # roughly correct, haven't consider in detail
    start_epoch = state.epoch + 1

    S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    steps_per_epoch = 0
    for i in utils.minibatch(users,
                        posItems,
                        negItems,
                        batch_size=world.config['bpr_batch_size']):
        steps_per_epoch += 1
    lr_scheduler  = torch.optim.lr_scheduler.OneCycleLR(bpr.opt, max_lr=world.config['lr'], epochs=world.TRAIN_epochs, steps_per_epoch=steps_per_epoch, last_epoch=start_epoch*steps_per_epoch-1, cycle_momentum=False)
    for epoch in range(start_epoch, start_epoch+world.TRAIN_epochs):
        start = time.time()
        if epoch % 5 == 0:
            cprint("[TEST]")
            results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            state.epoch = epoch-1
            state.save(results['ndcg'][0])
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, lr_scheduler, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
finally:
    if world.tensorboard:
        w.close()