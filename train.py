import pylearn2.datasets.cifar10 as cifar10

import model as Model
import util

if __name__ == "__main__":
	model = Model.Model('deconv-cifar10', '/experiments/deconv/cifar10/')
	util.load_checkpoint(model, '/experiments/cifar10/cifar10-09m-09d-01h-25m-08s.pkl')
	monitor = util.Monitor(model)

	train_dataset = cifar10.CIFAR10(which_set='train',
	                                rescale=True,
	                                axes=['c', 0, 1, 'b']
	                                )
	test_dataset = cifar10.CIFAR10(which_set='test',
	                               rescale=True,
	                               axes=['c', 0, 1, 'b']
	                               )
	train_iterator = train_dataset.iterator(mode='random_uniform', batch_size=128, num_batches=100000, topo=True)

	evaluator = util.Evaluator(model, test_dataset, steps=1000)

	for batch in train_iterator:
	    monitor.start()
	    error = model.train(batch)
	    monitor.stop(error)
	    evaluator.run()