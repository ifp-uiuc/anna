import numpy as np

class dataset_u(object):
	# Class to construct an unsupervised dataset

	def __init__(self, X):#, batch_size=None, iter_type=None):
		self.X = X
		self.n_samples = self.X.shape[0]
		self.n_channels = self.X.shape[1]
		self.height = self.X.shape[2]
		self.width = self.X.shape[3]	

		
	def __iter__(self):
		return self.iter #self.iterator()	

		
	def iterator(self, iter_mode=None, batch_size=None, num_batches=None, rng_seed=0):
		
		if(batch_size is not None):
			if(batch_size > self.n_samples):
				raise ValueError("batch_size is larger than number of samples!!")
		
		
	
		if(iter_mode is None or iter_mode == 'SEQ'):
					
			if batch_size is None:
				if num_batches is not None:
					batch_size = int(self.n_samples / num_batches)
				else:
					raise ValueError("need one of batch_size, num_batches"
									 "for sequential batch iteration")
									 
			elif batch_size is not None:
				if num_batches is not None:
					max_num_batches = int(np.ceil(self.n_samples / batch_size))
					if num_batches > max_num_batches:
						raise ValueError("dataset of %d examples can only provide "
										 "%d batches with batch_size %d, but %d "
										 "batches were requested" %
										(self.n_samples, max_num_batches,
										 batch_size, num_batches))
				else:
					num_batches = int(self.n_samples / batch_size)
		
		
			self.iter = Dataset_Iter_Seq(self.X, batch_size, num_batches, self.n_samples)	
			return self.iter
		elif(iter_mode == 'RAND_UNIF'):
			self.iter = Dataset_Iter_Rand_Uniform(self.X, batch_size, num_batches, self.n_samples, rng_seed)
			return self.iter
			
		elif(iter_mode == 'RAND_UNIF_nr'):
		
			if batch_size is None:
				if num_batches is not None:
					# Floor Function chosen to ensure that uneven segment (i.e. "runt") is ignored
					batch_size = int(np.floor(np.float32(self.n_samples) / num_batches))
				else:
					raise ValueError("need one of batch_size, num_batches"
									 "for sequential batch iteration")
									 
			elif batch_size is not None:
				if num_batches is not None:
					max_num_batches = int(np.ceil(self.n_samples / batch_size))
					if num_batches > max_num_batches:
						raise ValueError("dataset of %d examples can only provide "
										 "%d batches with batch_size %d, but %d "
										 "batches were requested" %
										(self.n_samples, max_num_batches,
										 batch_size, num_batches))
				else:
					# Floor Function chosen to ensure that uneven segment (i.e. "runt") is ignored
					num_batches = int(np.floor(np.float32(self.n_samples) / batch_size))
					
		
			self.iter = Dataset_Iter_Rand_Uniform_wo_replace(self.X, batch_size, num_batches, self.n_samples, rng_seed)
			return self.iter
			
	def get_num_samples(self):
		return self.n_samples


	def get_batch(self):	
		return self.iter.next()

        
class Basic_Iter(object):    

	def __init__(self, X, batch_size=None, num_batches=None, num_samples=None):

		self._batch_size  = batch_size
		self._num_batches = num_batches
		self._num_samples = num_samples
		print('Batch Size: {}'.format(batch_size))
		print('Num Batches: {}'.format(num_batches))
		print('Num Samples: {}'.format(num_samples))
				
		# Batch Counter
		self._batch_count = 0
		# Sample Counter
		self._sample_count = 0
		# Batch Indices
		self._last = 0
		
		
	def reset(self):
		self._batch_count = 0
		self._sample_count = 0
		self._last = 0
	
    
class Dataset_Iter_Seq(Basic_Iter):
	
	#
	# Iterator that traverses the data by extracting sequential slices
	# of size (batch_size) from the data tensor 
	#
	

	def __init__(self, X, batch_size=None, num_batches=None, num_samples=None):
		print('Using Sequential Iterator')
		super(Dataset_Iter_Seq, self).__init__(X, batch_size, num_batches, num_samples)
    

	def next(self):
		print('In Dataset_Iter_Seq_Next')
		if self._batch_count >= self._num_batches or self._sample_count >= self._num_samples:
			#print('If clause')
			raise StopIteration()
		# this fix the problem where (self._num_samples % self._batch_size) != 0
		elif (self._sample_count + self._batch_size) > self._num_samples:
			self._last = slice(self._sample_count, self._num_samples)
			self._sample_count = self._num_samples
			#return self._last
			return X[self._last, :, :, :]
		else:
			#print('Else Clause')
			self._last = slice(self._sample_count, self._sample_count + self._batch_size)
			self._sample_count += self._batch_size
			self._batch_count += 1
			#return self._last
			return X[self._last, :, :, :]
	
    
class Dataset_Iter_Rand_Uniform(Basic_Iter):

	#
	# Iterator that uniformly samples batches of size (batch_size)
	# from the data tensor (with replacement)
	#	

	def __init__(self, X, batch_size=None, num_batches=None, num_samples=None, rng_seed=0):
		print('Using Random Uniform Iterator (with replacement)')
		np.random.seed(rng_seed)
		super(Dataset_Iter_Rand_Uniform, self).__init__(X, batch_size, num_batches, num_samples)

		
		
	def next(self):
		if self._batch_count >= self._num_batches:
			raise StopIteration()
		else:
			self._last = np.random.random_integers(low=0,high=self._num_samples - 1,size=(self._batch_size,))
			#print(self._last)			
			self._batch_count += 1
			return X[self._last, :, :, :]

			
class Dataset_Iter_Rand_Uniform_wo_replace(Basic_Iter):

	#
	# Iterator that uniformly samples batches of size (batch_size)
	# from the data tensor (without replacement)
	#	

	def __init__(self, X, batch_size=None, num_batches=None, num_samples=None, rng_seed=0):
		print('Using Random Uniform Iterator (w/o replacement)') 
		np.random.seed(rng_seed)
		self._remain_samples = np.arange(num_samples)
		#print(_remain_samples)
		#g=raw_input('...')
		super(Dataset_Iter_Rand_Uniform_wo_replace, self).__init__(X, batch_size, num_batches, num_samples)

		
		
	def next(self):
		if self._batch_count >= self._num_batches:
			raise StopIteration()
		#elif len(self._remain_samples) < self._batch_size:
		#	print('In ELIF CLAUSE!!')
			
		else:
			_idx = np.random.permutation(np.arange(len(self._remain_samples)))[0:self._batch_size]
			self._last = self._remain_samples[_idx]
			self._remain_samples = np.setdiff1d(self._remain_samples, self._last)
			#print('_idx: {}'.format(_idx))
			#print('_last: {}'.format(self._last))
			#print('_remain_samples: {}'.format(self._remain_samples))
			#g=raw_input('...')
			
			self._batch_count += 1
			return X[self._last, :, :, :]			
			
			
if __name__ == "__main__":
	print('Making Datset Object')
	X = np.zeros((1024, 3, 5, 5))
	d = dataset_u(X)
	#NUM_SAMPLES = d.get_num_samples()
	#print('NUM SAMPLES: {}'.format(NUM_SAMPLES))
	
	d.iterator(batch_size=128)
	#d.iterator(iter_mode='RAND_UNIF', batch_size=128, num_batches = 1000, rng_seed=10)
	#d.iterator(iter_mode='RAND_UNIF_nr', batch_size=128)
	for i, b in enumerate(d):
		print('Batch {} --- {}'.format(i+1, b.shape))
		
