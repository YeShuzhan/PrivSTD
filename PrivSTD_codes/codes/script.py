import os

# eps = [0.1, 0.2, 0.5, 1.0, 2.0]
#
# for e in eps:
#     os.system(f'python train.py --eps {e}')

# window_size = [4, 8, 16]
# img_size = [64, 128, 256]
#
# for w in window_size:
#     for i in img_size:
#         if w == 8 and i == 128:
#             continue
#         os.system(f'python train.py --window_size {w} --img_size {i} --eps 0.5 --gpu 1')


# model = ['mv2s', 'sv']
#
# for m in model:
#     os.system(f'python train.py --model {m} --eps 0.5 --gpu 1')


sample_size = [128, 256]
for s in sample_size:
    os.system(f'python train.py --sample_size {s} --comment "grid size" --gpu 1')