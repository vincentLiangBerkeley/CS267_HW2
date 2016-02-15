#include "bin.h"

void init_grid(int num_bins, bin_t *bin_list)
{
    for(int i = 0; i < num_bins; i ++)
        bin_list[i].indeces = (int*)malloc(bin_list[i].capacity*sizeof(int));
}

// The list of bins will be separated into y*x
void set_grid_size(int &x, int &y, int num_bins)
{
    int m = static_cast<int>(sqrt(num_bins));
    for(int i=m; i>0; i --)
    {
        if(num_bins % i == 0) 
        {
            y = i;
            break;
        }
    }

    x = num_bins / y;
}

// The size of the global board is known from outside of this file
// Also, this function will also dynamically adjust the size of the bins
void bin_particles(int n, particle_t *particles, int num_bins, bin_t *bin_list, double bin_x, double bin_y, int num_rows)
{
    int i;
    for(i=0;i<num_bins;i++) bin_list[i].bin_size = 0;
    // Binning particles to bins
    for(i=0;i<n;i++)
    {   
        int x = particles[i].x / bin_x, y = particles[i].y / bin_y;
        int index = y+x*num_rows;
        if (bin_list[index].bin_size == bin_list[index].capacity)
            // Need to allocate more memory here
        {
            printf("Reallocating memory to bin # %d\n", index);
            bin_list[index].indeces = (int*) realloc(bin_list[index].indeces, 2*bin_list[index].capacity*sizeof(int));
            bin_list[index].capacity  *= 2;
        }
        bin_list[index].indeces[bin_list[index].bin_size] = i;
        bin_list[index].bin_size += 1;

        if (bin_list[index].bin_size > 20) printf("Warning more than 20 particles in this bin.\n");
    }
}

void sanity_check(int n, int num_bins, bin_t *bin_list)
{
    int sum = 0;
    for(int i = 0; i < num_bins; i ++)
    {
        // if(bin_list[i].bin_size > 2)
        //     printf("bin # %d has %d particles\n", i, bin_list[i].bin_size);
        sum += bin_list[i].bin_size;
    }
    
    if(sum == n) printf("The total number of particles is unchanged.\n");
    else printf("Sum = %d, n = %d\n", sum, n);
}