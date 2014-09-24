import argparse
from matplotlib import pyplot

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='log plotter', description='Script to plot log data')
    parser.add_argument('log_path', help='Path to log.txt')

    args = parser.parse_args()
    log_path = args.log_path

    file = open(log_path, 'r')
    lines = file.readlines()
    file.close()

    train_error = [float(line.split(',')[1].split(':')[1].split(' ')[1]) for line in lines if line.startswith('*')]
    test_error = [float(line.split(',')[1].split(':')[1].split(' ')[1]) for line in lines if line.startswith('&')] 

    pyplot.plot(train_error, label='Train')
    if test_error:
        pyplot.plot(test_error, label='Test')
    pyplot.xlabel('Steps / 50')
    pyplot.ylabel('Error')
    pyplot.legend()
    pyplot.show()
