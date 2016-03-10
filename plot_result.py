import sys, getopt
import re
import os
import matplotlib.pyplot as plt
from pprint import pprint

def plot_serial(filename, func):
    particles = []
    time = []
    lines = []
    with open(filename, 'rb') as f:
        lines = f.readlines()
    for line in lines:
        array = line.split(' ')
        particles.append(array[0])
        if func == 'serial':
            time.append(array[1])
        else:
            time.append(array[2])

    return (particles, time)

def plot_flops(filename):
    cores = []
    flops = []
    with open(filename, 'rb') as f:
        lines = f.readlines()
    for line in lines:
        array = line.split(' ')
        cores.append(array[0])
        flops.append(array[1])
    return (cores, flops)

def plot_strong(filename, func_name):
    linear = 0.0
    speedups = []
    cores = [1,2,4,6,8,12,24] if func_name == 'openmp' else [1,3,6,9,12,18,24]
    with open(filename, 'rb') as f:
        lines = f.readlines()
    for i in range(len(cores)):
        array = lines[i].split(' ')
        if int(array[1]) != cores[i]:
            print 'Error in file ' + filename
            sys.exit()
        if int(array[1]) == 1:
            linear = float(array[2])
            speedups.append(1.0)
        else:
            speedups.append(linear/float(array[2]))

    return (cores, speedups)


def main(argv):
    output_file = ''
    func_name = ''
    t = 'serial'
    r = 10000
    try:
        opts, args = getopt.getopt(argv, "hf:o:t:r:")
    except getopt.GetoptError:
        print "Usage: plot_result -f <func_name> -o <output_file> -t <type> -r <range>"
	sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
	    print "Usage: plot_result -f <func_name> -o <output_file> -t <type> -r <range>"
	    sys.exit()
    	elif opt in ('-f'):
    	    func_name = arg
    	elif opt in ('-o'):
    	    output_file = arg
    	elif opt in ('-t'):
    	    t = arg
    	elif opt in ('-r'):
    	    r = int(arg)
        else:
            print "Wrong Usage!"
            sys.exit()

    if t == 'serial':
        (particles, time) = plot_serial(output_file, func_name)
        plt.loglog(particles, time, basex=2, basey=2)
        plt.show()
    elif t == 'flops':
        (cores, flops) = plot_flops(output_file)
        plt.plot(cores, flops)
        plt.show()
    elif t == 'strong':
        cores, speedups = plot_strong(output_file, func_name)
        plt.bar(cores, speedups, width=0.35)
        plt.show()
    elif t == 'weak':
        job_units = []
        time = []
        if func_name == 'openmp':
            job_units = [1, 2, 3, 4, 6, 12]
            time = [0.406829, 0.520401, 0.597542, 0.678095, 0.719362, 0.851197]
        elif func_name == 'mpi':
            job_units = [1, 2, 3, 4]
            time = [1.03359, 1.19359, 1.25643, 1.26901]
        plt.bar(job_units, time, width = 0.35)
        plt.show()
    

if __name__ == "__main__":
    main(sys.argv[1:])
