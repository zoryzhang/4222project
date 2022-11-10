import LightGCN.code.world as world
import LightGCN.code.dataloader as dataloader
import LightGCN.code.model as model
import LightGCN.code.utils as utils
from pprint import pprint

if world.dataset in ['gowalla', 'yelp2018', 'amazon-book', 'movielens']:
    dataset = dataloader.Loader(path="LightGCN/data/"+world.dataset)
elif world.dataset == 'lastfm':
    dataset = dataloader.LastFM()

print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print("comment:", world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print("Test Topks:", world.topks)
print("using bpr loss")
print('===========end===================')

MODELS = {
    'mf': model.PureMF,
    'lgn': model.LightGCN
}