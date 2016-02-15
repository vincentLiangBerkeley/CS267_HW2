#ifndef __BIN_H__
#define __BIN_H__
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "common.h"
#include "bin.h"
typedef struct
{
    int* indeces;
    int bin_size;
    int capacity;
}bin_t;

// This function will allocate initial memory for each bin
void init_grid(int num_bins, bin_t *bin_list);
void set_grid_size(int &x, int &y, int num_bins);
// This function bins particles to bins, and also allocate memory dynamically when needed
void bin_particles(int n, particle_t *particles, int num_bins, bin_t *bin_list, double bin_x, double bin_y, int num_rows);
// Called only when debugging
void sanity_check(int n, int num_bins, bin_t *bin_list);
void clear_grid(int num_bins, bin_t *bin_list);
#endif