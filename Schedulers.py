class Schedulers:

	def __init__(self, initial_lrate, knee_epoch = 4):
		self.initial_lrate = initial_lrate
		self.knee_epoch = knee_epoch

	
	def lr_scheduler(self, epochNumber):
		lrate = self.initial_lrate
		if epochNumber == self.knee_epoch + 1:
			lrate = lrate * 0.1
		return lrate