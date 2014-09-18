"""Script to launch experiments.
"""
import argparse
import os
import shutil
import subprocess
import sys


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='launcher', description='Script to launch experiments')
    parser.add_argument('exp_name', help='Name of experiment')
    parser.add_argument('train_path', help='Path to model.py and train.py files')
    parser.add_argument('out_path', help='Path to to folder to save results')
    #parser.add_argument('-v', action='store_true', help='Verbose')
    args = parser.parse_args()

    print('\n========================================================')
    print('              Fastor Experiment Launcher                ')
    print('========================================================\n')    

    exp_name = args.exp_name
    train_path = args.train_path
    out_path = args.out_path

    print('===================== Inputs ===========================')
    print('Experiment Name: {}'.format(exp_name))
    print('Path to train.py and model.py: {}'.format(train_path))
    print('Out Folder Path: {}'.format(out_path))
    print('========================================================\n')

    # Modify output directory to include experiment name in path
    # Check if output directory exists, if not create it
    out_path = os.path.join(out_path, exp_name)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Copy model.py and train.py to out_file_path location
    model_file_path = os.path.join(train_path, 'model.py')
    train_file_path = os.path.join(train_path, 'train.py')

    if not os.path.exists(model_file_path):
        print('Model File Does Not Exist!')
        sys.exit(0)
    elif not os.path.exists(train_file_path):
        print('Train File Does Not Exist!')
        sys.exit(0)

    shutil.copy(model_file_path, os.path.join(out_path, 'model.py'))
    shutil.copy(train_file_path, os.path.join(out_path, 'train.py'))


    print('=================== Launch Process =======================')

    # Launch Process
    log_file = os.path.join(out_path, 'log.txt')
    f_log = open(log_file, 'wb')
    run_str = 'python {0} {1} {2} '.format(os.path.join(out_path, 'train.py'), exp_name, out_path)        
    run_list = run_str.split()
    print('Running: {}'.format(run_str))
    child_proc = subprocess.Popen(run_list, stdout=f_log, stderr=f_log)
    PID = child_proc.pid
    print('Process Launched with PID: {}'.format(PID))
    print('Process Log written to: {}'.format(log_file))

    # Make file containing the pid
    f_pid = open(os.path.join(out_path, 'pid'), 'wb')
    f_pid.write('PID: {}\n'.format(PID))
    f_pid.close()

    print('======================= Done ===========================')