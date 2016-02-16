import sys, getopt
import re
import os
import matplotlib.pyplot as plt
from pprint import pprint

def test_func(fun_name, r):
    os.system('make ' + fun_name)
    # The following data is for sequential performance
    res = []
    num_pars = []
    for i in xrange(1000, r, 1000):
        tmp = os.popen('./' + fun_name + ' -no -n '+str(i)).read()
        p = re.compile('(\d+.\d+)')
        outcome = p.findall(tmp)
        num_pars.append(int(outcome[0]))
        res.append(float(outcome[1]))
    return (num_pars, res)

def main(argv):
    output_file = ''
    func_name = ''
    plot = True
    r = 10000
    try:
        opts, args = getopt.getopt(argv, "hf:o:p:r:")
    except getopt.GetoptError:
        print "Usage: plot_result -f <func_name> -o <output_file> -p <plot> -r <range>"
	sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
	    print "Usage: plot_result -f <func_name> -o <output_file> -p <plot> -r <range>"
	    sys.exit()
	elif opt in ('-f'):
	    func_name = arg
	elif opt in ('-o'):
	    output_file = arg
	elif opt in ('-p'):
	    plot = arg
	elif opt in ('-r'):
	    r = int(arg)

    (num_pars, res) = test_func(func_name, r)
    if plot:
        plt.loglog(num_pars, res, basex=2, basey=2)
        plt.show()
    if output_file != '':
        target = open(output_file, 'w')
	target.write("Tested func: " + func_name + "\n")
	target.write(str(num_pars) + "\n")
	target.write(str(res))

    if func_name == 'openmp':
        # Plot speedup vs num_threads for openmp
        p = re.compile('(\d+.\d+)')
        serial = os.popen('./serial -no').read()
        serial_time = float(p.findall(serial)[1])
        speedup = []
        for i in xrange(1, 5):
            os.system('export OMP_NUM_THREADS=' + str(i))
            tmp = os.popen('./openmp -no').read()
            print tmp
            speedup.append(serial_time / float(p.findall(tmp)[1]))
        plt.bar(xrange(1, 5), speedup, 0.35)
        plt.show()     

    

if __name__ == "__main__":
    main(sys.argv[1:])
