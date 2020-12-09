gpu = 0

lamb = 10
epoch_count = 1
niter = 25
niter_decay = 25
lr = 0.0002
lr_policy = "lambda"
lr_decay_iters = 10
beta1 = 0.5
snap_period = 2500
save_ckpt_interval = 5000
write_log_interval = 100

# data
batch_size = 8
data_shape = [64, None]
data_dir = 'datasets/csnet_data'

train_data_dir = 'train'
test_data_dir = 'test'

i_s_dir = 'i_s'
mask_t_dir = 'mask_t'

example_data_dir = 'datasets/sample/train'
train_result_dir = 'result'

# train
train_ckpt_G_path = None
train_ckpt_D_path = None

# predict
predict_ckpt_path = 'weight/CSNet_weight.pth'
predict_data_dir = '/home/miyazonotaiga/デスクトップ/MyResearch/TGNet-Datagen/extract_and_recognize_title_region/result/title'
predict_result_dir = './'
gpu = False
