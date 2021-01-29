
import cv2
import glob, os, random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.activations import softmax, linear
from tensorflow.keras.layers import  Dense
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input



# Function to plot gradcam heatmap
def show_gradcam_heatmap(filename, model, labels, target_size):
    h = target_size[0]
    w = target_size[1]
        
    orig = cv2.imread(filename)
    resized = cv2.resize(orig, (h, w))
    
    test_image = load_img(filename, target_size=target_size)
    test_input = img_to_array(test_image)
    test_input = preprocess_input(test_input)
    test_input = np.expand_dims(test_input, axis=0)
    
    preds = model.predict(test_input)
    if len(preds) == 1:
        # Binary
        if (preds.ndim) > 1:
            preds = np.squeeze((preds),axis=1)
        idx = (preds > 0.5).astype(np.int)
        score = [round(score,3) if score>0.5 else round((1-score),3) for i,score in enumerate(preds)]

    else:
        #  Categorical
        idx = np.argmax(preds, axis=1)
        score = [round(preds[i,idx],3) for i,idx in enumerate(idx)]

    # Init GradCam class and select conv layer that we want to see its output
    cam = GradCAM(model, layerName='conv2d_3')
    
    # Compute heatmap
    heatmap = cam.compute_heatmap(test_input, idx)
    heatmap = cv2.resize(heatmap, (h, w))
        
    pred_label = labels[idx]
    pred_score = score
    
    rows = 1
    cols = 3
    fig_dims = (cols * 4, rows * 4)
    
    fig = plt.figure(figsize = fig_dims)
    fig.tight_layout(pad=2.0)
    
    plt.subplot(rows, cols, 1)
    plt.axis('off')
    plt.imshow(test_image)
    plt.title('Predicted: ' + pred_label + ' [' + str(round(pred_score*100)) + '%]')
    
    plt.subplot(rows, cols, 2)
    plt.axis('off')
    plt.imshow(heatmap, cmap='jet', alpha=0.85)
    
    plt.subplot(rows, cols, 3)
    plt.axis('off')
    test_image_gray = Image.open(filename).convert("LA")
    test_image_gray = test_image_gray.resize((w,h))
    plt.imshow(test_image_gray)
    plt.imshow(heatmap, cmap='jet', alpha=0.35)



class GradCAM:
    def __init__(self, model, classIdx, layerName):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # For layerName, it is recommended to choose the last conv layer of your model
        self.layerName = layerName

        # # if the layer name is None, attempt to automatically find the target output layer
        # if self.layerName is None:
        #     self.layerName = self.find_target_layer()
        
        # For very confident models, it is important to swap softmax 
        # activation by linear activation
        self.model = self.change_activation_layer(self.model)

    def find_target_layer(self):
    # attempt to find the final convolutional layer in the network
    # by looping over the layers of the network in reverse order
        
        for layer in reversed(self.model.layers):
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 4:
                return layer.name
            
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
    
    def change_activation_layer(self, model):
    # Swap activation of the last layer by a linear activation

        try:
            assert model.layers[-1].activation == softmax
        except:
            print('No softmax at the last layer')

        config = model.layers[-1].get_config()

        weights = [x.numpy() for x in model.layers[-1].weights]

        config['activation'] = linear
        config['name'] = 'logits'

        new_layer = Dense(**config)(model.layers[-2].output)
        new_model = Model(inputs=[model.input], outputs=[new_layer])
        new_model.layers[-1].set_weights(weights)

        new_config = new_model.layers[-1].get_config()

        try:
            assert new_model.layers[-1].activation == linear
        except:
            print('Linear activation has not be settled successfully')

        return new_model

    def compute_heatmap(self, image, eps=1e-8):
        
        gradModel = Model(inputs=[self.model.inputs],
                        outputs=[self.model.get_layer(self.layerName).output,self.model.output])
        
        with tf.GradientTape() as tape:

            # Watch the variable in order to calculate gradients
            tape.watch(self.gradModel.get_layer(self.layerName).output)

            # Cast image to tensor
            inputs = tf.cast(image, tf.float32)

            # Obtain output losses of the convolution layer and predictions
            (convOutputs, predictions) = self.gradModel(inputs)

            if len(predictions)==1:
                # Binary
                loss = predictions[0]
            else:
                # Categorical 
                loss = predictions[:, self.classIdx]
        
        # Calculate gradient of the loss of the predicted class  w.r.t convOutputs
        grads = tape.gradient(loss, convOutputs)

        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads

        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]

        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

        # grab the spatial dimensions of the input image and resize
        # the output class activation map to match the input image
        # dimensions
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))

        # normalize the heatmap such that all values lie in the range
        # [0, 1], scale the resulting values to the range [0, 255],
        # and then convert to an unsigned 8-bit integer
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")

        return heatmap

    def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_VIRIDIS):
        heatmap = cv2.applyColorMap(heatmap, colormap)
        output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

        return (heatmap, output)


if __name__ == "__main__":
    model_path = 'model/my_model.h5'
    test_files = glob.glob('images/*.png')
    
    labels = ["NOK", "OK"]
    num_classes = len(labels)
    target_size = 224
    layer_name = 'conv2d_3'

    model = load_model(model_path)

    if num_classes == 2:
        model.compile(loss="binary_crossentropy",
                    optimizer=optimizers.Adam(lr=1e-03),
                    metrics=["acc"])
    else:
        model.compile(loss="categorical_crossentropy",
                optimizer=optimizers.Adam(lr=1e-03),
                metrics=["acc"])

    random.shuffle(test_files)
    for filename in test_files:
        show_gradcam_heatmap(filename, labels, model, target_size, layer_name=layer_name)


