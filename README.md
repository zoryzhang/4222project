# 4222project
Graph-Learning-Based Recommender System on MovieLens

Our project is built highly based on https://github.com/microsoft/recommenders and https://github.com/gusye1234/LightGCN-PyTorch (Thank you!)

Project Title : BiasedGCN
Made to work with Recommender Systems
Done in correlation with a Presentation and a Report
For Course COMP 4222 : Machine Learning with Strcutured Data in Fall 2022-23 at HKUST

Made by AGARWAL, Sahil ; WEI, Yuanjing ; ZHANG, Yujun

# Run
For the original stacking, we apply the same training setting as suggested in the paper.
`CUDA_VISIBLE_DEVICES=1 python LightGCN2/code/main.py --decay=1e-4 --lr=0.001 --recdim=64 --bpr_batch=2048 --load=0 --stacking_func=0 --seed=2031 --epochs=500 --comment=sf0_decay4_lr3_bt2048`

`CUDA_VISIBLE_DEVICES=2 python LightGCN2/code/main.py --decay=1e-3 --lr=1e-3 --recdim=64 --bpr_batch=2048 --load=0 --stacking_func=1 --seed=2029 --epochs=500 --comment=sf1_decay3_lr3_bt2048`

`CUDA_VISIBLE_DEVICES=3 python LightGCN2/code/main.py --decay=1e-2 --lr=1e-3 --recdim=64 --bpr_batch=2048 --load=0 --stacking_func=3 --seed=2030 --epochs=500 --comment=sf3_decay2_lr3_bt2048`