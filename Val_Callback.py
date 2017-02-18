import keras
import DiceMetric as DM

class Val_Callback(keras.callbacks.Callback):
    def __init__(self, val_data, train_data, model):
        self.val_data = val_data
        self.train_data = train_data
        self.model = model
	
	
    def on_epoch_end(self, epoch, logs={}):
        x_val, y_val = self.val_data
        x_train, y_train = self.train_data

        dice_score_validation = self.model.evaluate(x = x_val , y = y_val, verbose = 0)
        dice_score_training = self.model.evaluate(x = x_train, y = y_train, verbose = 0)

        print('\nValidation Dice Score: {}  Training Dice Score: {}\n'.format(dice_score_validation[1], dice_score_training[1]))
	
