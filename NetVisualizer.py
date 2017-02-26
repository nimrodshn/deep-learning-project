import keras.backend as K
import matplotlib
import matplotlib.pyplot as plt
from scipy.misc import imsave
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma
import numpy as np

def plot_activations(model, layer_num):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    
    filter_index = 3 #always plotting result of first filter.
    img_height = 64
    img_width = 64
    
    inspected_layer = model.layers[layer_num-1]
    input_image = inspected_layer.input
    
    layer_output = layer_dict[inspected_layer.name].output
    loss = K.mean(layer_output[:, filter_index, :, :])
    
    grads = K.gradients(loss, input_image)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    iterate = K.function([input_image], [loss, grads])
    
    input_img_data = np.random.random((1, img_width, img_height, 1)) * 20 + 128.

    for step in range(20):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    
    img = input_img_data[0]
    img = deprocess_image(img).squeeze()
    
    fig=plt.figure()
    plt.imshow(img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

    #imsave('%s_filter_%d.png' % (inspected_layer.name , filter_index) , img)	

def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def plot_feature_map(model, layer_id, X, n=256, ax=None, **kwargs):

    layer = model.layers[layer_id]
    try:
        get_activations = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
        activations = get_activations([X, 0])[0]
    except:
        # Ugly catch, a cleaner logic is welcome here.
        raise Exception("This layer cannot be plotted.")

    # For now we only handle feature map with 4 dimensions
    if activations.ndim != 4:
        raise Exception("Feature map of '{}' has {} dimensions which is not supported.".format(layer.name,
            activations.ndim))

        # Set default matplotlib parameters
    if not 'interpolation' in kwargs.keys():
        kwargs['interpolation'] = "none"

    if not 'cmap' in kwargs.keys():
        kwargs['cmap'] = "gray"

    fig = plt.figure(figsize=(15, 15))

    # Compute nrows and ncols for images
    n_mosaic = len(activations)
    nrows = int(np.round(np.sqrt(n_mosaic)))
    ncols = int(nrows)
    if (nrows ** 2) < n_mosaic:
        ncols +=1

    # Compute nrows and ncols for mosaics
    if activations[0].shape[0] < n:
        n = activations[0].shape[0]

    nrows_inside_mosaic = int(np.round(np.sqrt(n)))
    ncols_inside_mosaic = int(nrows_inside_mosaic)

    if nrows_inside_mosaic ** 2 < n:
        ncols_inside_mosaic += 1

    for i, feature_map in enumerate(activations):

        mosaic = make_mosaic(feature_map[:n], nrows_inside_mosaic, ncols_inside_mosaic, border=1)

        ax = fig.add_subplot(nrows, ncols, i+1)

        im = ax.imshow(mosaic, **kwargs)
        ax.set_title("Feature map #{} \nof layer#{} \ncalled '{}' \nof type {} ".format(i, layer_id,
            layer.name,
            layer.__class__.__name__))

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)

    fig.tight_layout()
    return fig


def plot_all_feature_maps(model, X, n=256, ax=None, **kwargs):

    figs = []

    for i, layer in enumerate(model.layers):
        fig = plot_feature_map(model, i, X, n=n, ax=ax, **kwargs)
        figs.append(fig)
    return figs

def make_mosaic(im, nrows, ncols, border=1):
    """From http://nbviewer.jupyter.org/github/julienr/ipynb_playground/blob/master/keras/convmnist/keras_cnn_mnist.ipynb
    """

    nimgs = len(im)
    imshape = im[0].shape

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
        ncols * imshape[1] + (ncols - 1) * border),
        dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    im
    for i in range(nimgs):

        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0],
                col * paddedw:col * paddedw + imshape[1]] = im[i]
        return mosaic


def visualize_model_history(history):
    fig=plt.figure()
    print(history.history)
    #summarize history for accuracy
    plt.plot(history.history['dice_coeff'])
    plt.plot(history.history['val_dice_coeff'])
    plt.title('model accuracy (measured by dice coefficiant)')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
	
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_filters(layer, x ,y):
    filters = layer.get_value()
    fig = plt.figure()
    for idx, layer in enumerate(filters):
        ax = fig.add_subplot(x,y, i+1)
        ax.matshow(filters[idx][0], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
    plt.tight_layout()
    return plt
