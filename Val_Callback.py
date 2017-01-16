import keras
class Val_Callback(keras.callbacks.Callback):
    def __init__(self, val_data, model):
        self.val_data = val_data
        self.model = model

    def on_batch_end(self, batch, logs={}):
        x, y = self.val_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nValidation loss: {}, acc: {}\n'.format(loss, acc))
