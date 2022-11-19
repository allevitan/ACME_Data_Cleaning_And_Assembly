"""A script to process .stxm files to .cxi files

Author: Abe Levitan, alevitan@mit.edu
"""
import sys
import argparse
import numpy as np
import torch as t
from matplotlib import pyplot as plt
import cdtools
from cdtools.tools import plotting as p

# The shear calculations are so fast, there's no point in doing them
# on the GPU
default_shear = np.array([[ 0.99961877, -0.06551266],
                          [ 0.02651655,  0.99879594]])

def view_file(cxi_file, logarithmic=True):
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(cxi_file, cut_zeros=False)
    dataset.inspect()
    mean_pat = t.mean(dataset.patterns,axis=0)
    p.plot_real(mean_pat)
    plt.title('Mean diffraction pattern')
    plt.figure()
    delta = 1
    #plt.semilogy(t.sqrt((mean_pat[479] + delta)**2-delta**2))
    plt.semilogy(mean_pat[479])
    plt.semilogy(mean_pat[480])
    plt.show()

def main(argv=sys.argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('cxi_file', type=str, help='The file to view')
    parser.add_argument('--linear', action='store_true', help='If set, view the patterns on a linear rather than logarithmic scale.')
    
    args = parser.parse_args()

    view_file(args.cxi_file, logarithmic=~args.linear)


if __name__ == '__main__':
    sys.exit(main())

