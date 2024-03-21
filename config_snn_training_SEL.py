'''
    Configuration for SNN direct training

'''

# GPU setting
import os
#os.environ['NCCL_P2P_DISABLE']='1'
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="6"

#
from config import config
conf = config.flags

#
# conf.debug_mode = True
#conf.verbose_snn_train = True


#
# conf.mode='inference'
#conf.batch_size_inf=100
#conf.batch_size=400
#conf.batch_size=200
#conf.batch_size=300
#conf.batch_size=180
#conf.batch_size=120
#conf.time_step=2
# conf.name_model_load='./models_ckpt_SEL'
# conf.name_model_load='./models_ckpt_SEL_adaptive_0_8/VGG16_AP_CIFAR10/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s'
# conf.name_model_load='./models_ckpt_SEL_adaptive_0_8/VGG16_AP_CIFAR10/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-psp-ma-0.001_1e-10'
conf.name_model_load='./models_ckpt_SEL_adaptive_0_9/VGG16_AP_CIFAR10/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-psp-ma-0.001_1e-10'
# conf.name_model_load='./models_ckpt_SEL_loss/VGG16_AP_CIFAR10/ep-300_bat-100_opt-SGD_lr-STEP-1E-01_lmb-1E-04_sc_cm_ts-4_nc-R-R_nr-s_r-psp-ma-0.001_1e-10'


#
#conf.learning_rate = 0.2
#conf.lmb = 1E-4
#conf.time_step = 4
#conf.optimizer = 'ADAM'
#conf.lr_schedule = None

#
#conf.train_epoch = 150
#conf.step_decay_epoch = 50
#conf.train_epoch = 10
#conf.train_epoch = 10
#conf.num_train_data = 10000

#conf.model='VGG11'
conf.model='VGG16'
#conf.model='ResNet20'
#conf.model='ResNet32'

#conf.dataset='CIFAR100'
#conf.dataset='ImageNet'
#conf.dataset='CIFAR10_DVS'


conf.pooling_vgg = 'avg'

conf.nn_mode = 'SNN'
# conf.nn_mode = 'ANN'

conf.n_reset_type = 'reset_by_sub'
#conf.n_reset_type = 'reset_to_zero'


conf.vth_rand_static = False
conf.vrest = 0.0
#conf.vrest_rand_static = False
#conf.vrest_rand_static = True

# conf.adaptive_vth_SEL = False
conf.adaptive_vth_SEL = True
conf.adaptive_dec_vth_scale = 0.7
conf.adaptive_inc_vth_scale = 1.1

#
conf.reg_psp=True
conf.reg_psp_SEL = 1

#conf.use_bn=False

#conf.n_init_vth = 0.3

conf.leak_const_init = 0.9
#conf.leak_const_train = True


#
if False:
    #if True:
    if True:
        #if False:
        conf.reg_spike_out=True
        conf.reg_spike_out_const=2.75E-5
        conf.reg_spike_out_alpha=4
        conf.reg_spike_out_sc=True
        #conf.reg_spike_out_sc_wta=False
        #conf.reg_spike_out_sc_train=True
        conf.reg_spike_out_sc_sm=True
        #conf.reg_spike_out_sc_sq=True
        conf.reg_spike_out_norm=True

        #
        #conf.reg_psp=True
        conf.reg_psp_const=1E-3
        conf.reg_psp_eps=1E-10
        conf.reg_psp_min=True
    else:
        conf.reg_spike_out=True
        conf.reg_spike_out_const=5E-5
        conf.reg_spike_out_alpha=0
        #conf.reg_spike_out_sc=True
        #conf.reg_spike_out_sc_wta=False
        #conf.reg_spike_out_sc_train=True
        #conf.reg_spike_out_sc_sm=True
        #conf.reg_spike_out_sc_sq=True
        conf.reg_spike_out_norm=True

        #
        #conf.reg_psp=True
        conf.reg_psp_const=1E-3
        conf.reg_psp_eps=1E-10
        conf.reg_psp_min=True


#
#conf.grad_clipnorm = 3.0
#conf.grad_clipnorm = 1.0

#
#conf.en_stdp_pathway = True
conf.stdp_pathway_weight = 0.1

#
conf.save_best_model_only = True
conf.save_models_max_to_keep = 1
conf.root_model_save = './models_ckpt_SEL_adaptive_0_7_reg'
# conf.root_model_load = './models_ckpt_normal'


#
config.set()
