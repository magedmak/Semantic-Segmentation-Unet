from keras.utils.vis_utils import plot_model


def get_model_arch(model):
    img_file = './output/model_arch.png'
    plot_model(model, to_file=img_file, show_shapes=True, show_layer_names=True)

    