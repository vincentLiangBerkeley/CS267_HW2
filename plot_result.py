import sys, getopt
import re
import os
import matplotlib.pyplot as plt
from pprint import pprint

def test_func(fun_name, r):
    if fun_name == 'serial':
        os.system('make serial')
	res = []
	num_pars = []
	for i in xrange(1000, r, 1000):
	    tmp = os.popen('./serial -no -n '+str(i)).read()
	    p = re.compile('(\d+.\d+)')
	    outcome = p.findall(tmp)
	    num_pars.append(outcome[0])
	    res.append(outcome[1])
	
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
        plt.plot(num_pars, res)
	plt.show()
    if output_file != '':
        target = open(output_file, 'w')
	target.write("Tested func: " + func_name + "\n")
	target.write(str(num_pars) + "\n")
	target.write(str(res))
    

if __name__ == "__main__":
    main(sys.argv[1:])
