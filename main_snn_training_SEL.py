
#
# configuration
from config_snn_training_SEL import config


# snn library
import lib_snn

#
import datasets
import callbacks

########################################
# configuration
########################################
dist_strategy = lib_snn.utils.set_gpu()


################
# name set
################
#
filepath_save, filepath_load, config_name = lib_snn.utils.set_file_path()

########################################
# load dataset
########################################
train_ds, valid_ds, test_ds, train_ds_num, valid_ds_num, test_ds_num, num_class, train_steps_per_epoch = \
    datasets.datasets.load()
#datasets.datasets_bck_eventdata.load()


#
with dist_strategy.scope():

    ########################################
    # build model
    ########################################
    #data_batch = valid_ds.take(1)
    #model = lib_snn.model_builder.model_builder(num_class,train_steps_per_epoch)
    model = lib_snn.model_builder.model_builder(num_class,train_steps_per_epoch,valid_ds)

    ########################################
    # load model
    ########################################
    if config.load_model:
        model.load_weights(config.load_weight)

    ################
    # Callbacks
    ################
    callbacks_train, callbacks_test = \
        callbacks.callbacks_snn_train(model,train_ds_num,valid_ds,test_ds_num)

    #
    if True:
    # if False:
        if config.train:
            print('Train mode')

            model.summary()
            #train_steps_per_epoch = train_ds_num/batch_size
            train_epoch = config.flags.train_epoch
            init_epoch = config.init_epoch
            train_histories = model.fit(train_ds, epochs=train_epoch, steps_per_epoch=train_steps_per_epoch,
                                        initial_epoch=init_epoch, validation_data=valid_ds, callbacks=callbacks_train)
        else:
            print('Test mode')

            result = model.evaluate(test_ds, callbacks=callbacks_test)

# visualization - feature map - spike count
    if False:
    # if True:
        import tensorflow as tf
        import keras
        import matplotlib.pyplot as plt
        import numpy as np


        result = model.evaluate(test_ds.take(1), callbacks=callbacks_test)
        #result = model.evaluate(test_ds.take(10), callbacks=callbacks_test)
        #result = model.evaluate(test_ds, callbacks=callbacks_test)


        # move to prop.py postproc_batch_test()
        if True:
        # if False:
            fm = []
            layer_names = []

            for layer in model.layers_w_neuron:
                if isinstance(layer.act,lib_snn.neurons.Neuron):
                    fm.append(layer.act.spike_count_int)
                    layer_names.append(layer.name)

            #a = fm[13]
            #plt.matshow(a[0,:,:,0])

            images_per_row = 16

            img_idx = 0
            layer_idx = 0

            #display_grid_h = np.zeros((4,4))
            #plt.figure(figsize=(display_grid_h.shape[1], display_grid_h.shape[0]))

            plot_hist = False
            if plot_hist:
                figs_h, axes_h = plt.subplots(4, 4, figsize=(12,10))

            # only conv1
            fm = [fm[1]]
            layer_names = [layer_names[1]]

            #
            for i in range(0,1):
                result = model.evaluate(test_ds.skip(i).take(1), callbacks=callbacks_test)
                for img_idx in range(20, 23):
                    for layer_name, layer_fm in zip(layer_names,fm):
                        n_features = layer_fm.shape[-1]
                        size = layer_fm.shape[1]

                        n_cols = n_features // images_per_row
                        if n_cols == 0: # n_in
                            continue

                        if len(layer_fm.shape) == 2:    # fc_layers
                            continue

                        display_grid = np.zeros(((size+1)*n_cols-1,images_per_row*(size+1)-1))

                        #
                        for col in range(n_cols):
                            for row in range(images_per_row):
                                #
                                channel_index = col * images_per_row + row
                                channel_image = layer_fm[img_idx,:,:,channel_index].numpy().copy()

                                # normalization
                                if channel_image.sum() != 0:
                                    channel_image -= channel_image.mean()
                                    channel_image /= channel_image.std()
                                    channel_image *= 64
                                    channel_image += 128
                                channel_image = np.clip(channel_image,0,255).astype("uint8")

                                display_grid[
                                    col * (size+1):(col+1)*size + col,
                                    row * (size+1):(row+1)*size + row] = channel_image

                        #
                        scale = 1./size
                        plt.figure(figsize=(scale*display_grid.shape[1],
                                            scale*display_grid.shape[0]))
                        plt.title(layer_name)
                        plt.grid(False)
                        plt.axis("off")

                        plt.imshow(display_grid, aspect="auto", cmap="viridis")
                        plt.show()


                        # channel intensity
                        # image
                        stat_image= False
                        if stat_image:
                            # one image
                            channel_image = layer_fm[img_idx,:,:,:].numpy().copy()
                        else:
                            ## batch
                            channel_image = layer_fm.numpy().copy()

                        channel_intensity = tf.reduce_mean(channel_image,axis=[0,1])

                        #display_grid_h[layer_idx//4, layer_idx%4] = channel_intensity

                        if plot_hist:
                            axes_h[layer_idx//4,layer_idx%4].hist(tf.reshape(channel_intensity,shape=-1),bins=100)

                            ci_mean = tf.reduce_mean(channel_intensity)
                            ci_max = tf.reduce_max(channel_intensity)
                            ci_min = tf.reduce_min(channel_intensity)
                            ci_std = tf.math.reduce_std(channel_intensity)
                            ci_non_zeros = tf.math.count_nonzero(channel_intensity,dtype=tf.int32)
                            ci_non_zeros_r = ci_non_zeros/tf.math.reduce_prod(channel_intensity.shape)

                            print("{:}, mean:{:.3f}, max:{:.3f}, min:{:.3f}, std:{:.4f}, nonz:{:.3f}"
                                  .format(layer_fm.name,ci_mean,ci_max,ci_min,ci_std,ci_non_zeros_r))

                            layer_idx += 1

                    # fname = '_'+str(i)+'_'+str(img_idx)+'.png'
                    # folder = './result_fig_fm_sc_0050/'
                    # try:
                    #     if not os.path.exists(folder):
                    #         os.makedirs(folder)
                    # except:
                        # print("Error: failed to create the directory")
                    # plt.savefig(folder+fname)
                    plt.close()