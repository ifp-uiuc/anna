import os
import argparse
from matplotlib import pyplot

def get_error(log_path, char):
	file = open(log_path, 'r')
	lines = file.readlines()
	file.close()

	error = [float(line.split(',')[1].split(':')[1].split(' ')[1]) for line in lines if line.startswith(char)]
	return error

if __name__ == "__main__":
	parser = argparse.ArgumentParser(prog='log plotter', description='Script to plot log data')
	parser.add_argument('log_path', help='Path to log.txt')
	parser.add_argument('char', help='Character to plot (*=train, &=test)')

	args = parser.parse_args()
	log_path = args.log_path
	char = args.char

	files = [os.path.join(log_path, file) for file in os.listdir(log_path) if file.endswith('.txt')]

	files.sort()

	for file in files:
		filename = os.path.basename(file)
		label = filename.split('.')[0]
		error = get_error(file, char)
		pyplot.plot(error, label=label)
	pyplot.legend()
	pyplot.show()
