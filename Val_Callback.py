import keras
import DiceMetric as DM


class Val_Callback(keras.callbacks.Callback):
    def __init__(self, val_data, model):
        self.val_data = val_data
        self.model = model
	
	
    def on_epoch_end(self, epoch, logs={}):
        x, y = self.val_data
        #loss, acc = self.model.evaluate(x, y, verbose=0)
        predictions = self.model.predict(x)
        dice_score = DM.np_dice_coeff(y, predictions)
        print('\nValidation Dice Score: {}\n'.format(dice_score))
	