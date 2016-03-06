#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <cuda.h>
#include "common.h"

#define NUM_THREADS 256
#define DEBUG 0

extern double size;
__device__ int location_particle(particle_t &particle, double column_step, double row_step, int num_columns_bins);
__global__ void init_bin_list(int num_bins, int *bin_list);
__global__ void init_particle_list(int n, particle_t* particles, int* particle_list, int* bin_list, double column_step, double row_step, int num_columns_bins);
void set_bin_size(int &num_columns_bins, int &num_rows_bins, int num_bins);
__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor);
__device__ void apply_force_gpu_self_bin(int tid, particle_t* particles, int* particle_list, int last_particle_in_bin);
__device__ void apply_force_gpu_other_bin(int tid, particle_t* particles, int* particle_list, int last_particle_in_bin);
__global__ void compute_forces_gpu(particle_t* particles, int n, int num_rows_bins, int num_columns_bins, int* particle_list, int* bin_list, double column_step, double row_step);
__global__ void move_gpu (particle_t * particles, int n, double size);

//
// Find the index of bin where a given particle lives
//

// The index is like following
//  16,17,18,19,20
// 11,12,13,14,15
// 6,7,8,9,10
// 1,2,3,4,5 where num_columns_bins is 5, num_rows_bins is 4

__device__ int location_particle(particle_t &particle, double column_step, double row_step, int num_columns_bins)
{
    int column = particle.x / column_step;
    int row = particle.y / row_step;
    return column + row * num_columns_bins;
}

//
// Initialize bin_list
// bin_list[k] = -1 means no point enter kth bin yet
//
__global__ void init_bin_list(int num_bins, int *bin_list)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= num_bins) return;

    bin_list[tid] = -1;
}

//
// Initialize particle_list
// will also be used to assign particles to bins after each time step
//
__global__ void init_particle_list(int n, particle_t* particles, int* particle_list, int* bin_list, double column_step, double row_step, int num_columns_bins)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;

    particle_t particle = particles[tid];
    int bin_num = location_particle(particle, column_step, row_step, num_columns_bins);
    particle_list[tid] = atomicExch(&bin_list[bin_num], tid);
}

//
// For given num_bins, set balanced num_columns_bins and num_row_bins such that num_columns_bins * num_rows_bins == num_bins
//
void set_bin_size(int &num_columns_bins, int &num_rows_bins, int num_bins)
{
    int m = static_cast<int>(sqrt(num_bins));
    for( int i = m; i > 0; --i )
    {
        if(num_bins % i == 0)
        {
            num_rows_bins = i;
            break;
        }
    }
    num_columns_bins = num_bins / num_rows_bins;
}

//
//  compute force caused by neighbor on target particle
//
__device__ void apply_force_gpu(particle_t &particle, particle_t &neighbor)
{
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if( r2 > cutoff*cutoff )
        return;
    //r2 = fmax( r2, min_r*min_r );
    r2 = (r2 > min_r*min_r) ? r2 : min_r*min_r;
    double r = sqrt( r2 );

    //
    //  very simple short-range repulsive force
    //
    double coef = ( 1 - cutoff / r ) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

//
// compute the force caused by other particles in the same bin as given particle
//
__device__ void apply_force_gpu_self_bin(int tid, particle_t* particles, int* particle_list, int last_particle_in_bin)
{
    particle_t* p = &particles[tid];
    int i = last_particle_in_bin; // return the index of the last particle enters bin with index bin_index
    while (i != -1)
    {
        if (i != tid) // not interact with self
            apply_force_gpu(*p, particles[i]);
        i = particle_list[i];
    }
}

//
// compute the force caused by other particles in neighbor bin
//
__device__ void apply_force_gpu_other_bin(int tid, particle_t* particles, int* particle_list, int last_particle_in_bin)
{
    particle_t* p = &particles[tid];
    int i = last_particle_in_bin;
    while (i != -1)
    {
        apply_force_gpu(*p, particles[i]);
        i = particle_list[i];
    }
}

//
//  loop over every particle, find the bin where the particle lives
//  then consider the force caused by that bin and the neighbor bins
//  at most 8 bins
//

//
// particle_t stores the initial address of all particles
// n is the number of particles
// num_rows_bins is the number of rows in the grid
// num_columns_grid is the number of column in the grid
// every point in the grid represent for a block of threads
//

__global__ void compute_forces_gpu(particle_t* particles, int n, int num_rows_bins, int num_columns_bins, int* particle_list, int* bin_list, double column_step, double row_step)
{
    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) return;

    particles[tid].ax = particles[tid].ay = 0;
    int bin_num = location_particle(particles[tid], column_step, row_step, num_columns_bins);
    int column_num = particles[tid].x / column_step;
    int row_num = particles[tid].y / row_step;
    apply_force_gpu_self_bin(tid, particles, particle_list, bin_list[bin_num]);
    if (column_num > 0) // left neighbor bin
    {
        apply_force_gpu_other_bin(tid, particles, particle_list, bin_list[bin_num-1]);
        if (row_num > 0) // left lower neighbor bin
            apply_force_gpu_other_bin(tid, particles, particle_list, bin_list[bin_num-1-num_columns_bins]);
        if (row_num < num_rows_bins) // left upper neighbor bin
            apply_force_gpu_other_bin(tid, particles, particle_list, bin_list[bin_num-1+num_columns_bins]);
    }
    if (column_num < num_columns_bins) // right neighbor bin
    {
        apply_force_gpu_other_bin(tid, particles, particle_list, bin_list[bin_num+1]);
        if (row_num > 0) // right lower neighbor bin
            apply_force_gpu_other_bin(tid, particles, particle_list, bin_list[bin_num+1-num_columns_bins]);
        if (row_num < num_rows_bins) //right upper neighbor bin
            apply_force_gpu_other_bin(tid, particles, particle_list, bin_list[bin_num+1+num_columns_bins]);
    }
    if (row_num > 0) // lower neighbor bin
        apply_force_gpu_other_bin(tid, particles, particle_list, bin_list[bin_num-num_columns_bins]);
    if (row_num < num_rows_bins) // upper neighbor bin
        apply_force_gpu_other_bin(tid, particles, particle_list, bin_list[bin_num+num_columns_bins]);
}

__global__ void move_gpu (particle_t * particles, int n, double size)
{

    // Get thread (particle) ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= n) return;

    particle_t * p = &particles[tid];
    //
    //  slightly simplified Velocity Verlet integration
    //  conserves energy better than explicit Euler method
    //
    p->vx += p->ax * dt;
    p->vy += p->ay * dt;
    p->x  += p->vx * dt;
    p->y  += p->vy * dt;

    //
    //  bounce from walls
    //
    while( p->x < 0 || p->x > size )
    {
        p->x  = p->x < 0 ? -(p->x) : 2*size-p->x;
        p->vx = -(p->vx);
    }
    while( p->y < 0 || p->y > size )
    {
        p->y  = p->y < 0 ? -(p->y) : 2*size-p->y;
        p->vy = -(p->vy);
    }

}

int main( int argc, char **argv )
{
    // This takes a few seconds to initialize the runtime
    cudaThreadSynchronize();

    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        return 0;
    }

    int n = read_int( argc, argv, "-n", 1000 );

    char *savename = read_string( argc, argv, "-o", NULL );

    FILE *fsave = savename ? fopen( savename, "w" ) : NULL;
    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );

    // GPU particle data structure
    particle_t *d_particles;
    cudaMalloc((void **) &d_particles, n * sizeof(particle_t));

    set_size( n );

    init_particles( n, particles );

    cudaThreadSynchronize();
    double copy_time = read_timer( );

    // Copy the particles to the GPU
    cudaMemcpy(d_particles, particles, n * sizeof(particle_t), cudaMemcpyHostToDevice);

    cudaThreadSynchronize();
    copy_time = read_timer( ) - copy_time;

    // compute number of bins
    int num_bins = n % 4 == 0 ? n/4:n/4+1;
    // compute num_columns_bins and num_rows_bins such that num_columns_bins * num_rows_bins == num_bins
    int num_columns_bins, num_rows_bins;
    set_bin_size(num_columns_bins, num_rows_bins, num_bins);
    //if (DEBUG)
    //    printf("num_columns_bins is %d, num_rows_bins is %d, num_bins is %d. \n", num_columns_bins, num_rows_bins, num_bins);
    double column_step = size / num_columns_bins;
    double row_step = size / num_rows_bins;
    //if (DEBUG)
    //    printf("column_step is %lf, row_step is %lf", column_step, row_step);

    // particle_list is a int array with size n
    // bin_list is a int array with size num_bins
    // Initialize bin_list and particle_list in device
    int * particle_list;
    int * bin_list;
    int bins_blks = (num_bins + NUM_THREADS - 1) / NUM_THREADS;
    int blks = (n + NUM_THREADS - 1) / NUM_THREADS;
    cudaMalloc((void **) &particle_list, n * sizeof(int));
    cudaMalloc((void **) &bin_list, num_bins * sizeof(int));

    cudaThreadSynchronize();
    // bin_list will initialized to be full of -1, means no particles assigned to bin yet
    init_bin_list <<< bins_blks, NUM_THREADS >>> (num_bins, bin_list);
    init_particle_list <<< blks, NUM_THREADS >>> (n, d_particles, particle_list, bin_list, column_step, row_step, num_columns_bins);

    //
    //  simulate a number of time steps
    //
    cudaThreadSynchronize();
    double simulation_time = read_timer( );

    for( int step = 0; step < NSTEPS; step++ )
    {
        //if (DEBUG)
        //    printf("step = %d\n", step);
        //
        //  compute forces
        //
        compute_forces_gpu <<< blks, NUM_THREADS >>> (d_particles, n, num_rows_bins, num_columns_bins, particle_list, bin_list, column_step, row_step);
        //if (DEBUG)
        //    printf("compute_forces_gpu\n");
        //
        //  move particles
        //
        move_gpu <<< blks, NUM_THREADS >>> (d_particles, n, size);
        //if (DEBUG)
        //    printf("move_gpu\n");

        // reassign particles to bins
        init_bin_list <<< bins_blks, NUM_THREADS >>> (num_bins, bin_list);
        //if (DEBUG)
        //    printf("init_bin_list\n");
        init_particle_list <<< blks, NUM_THREADS >>> (n, d_particles, particle_list, bin_list, column_step, row_step, num_columns_bins);
        //if (DEBUG)
        //    printf("init_particle_list\n");

        //
        //  save if necessary
        //
        if( fsave && (step%SAVEFREQ) == 0 )
        {
            // Copy the particles back to the CPU
            cudaMemcpy(particles, d_particles, n * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save( fsave, n, particles);
        }
    }
    cudaThreadSynchronize();
    simulation_time = read_timer( ) - simulation_time;

    printf( "CPU-GPU copy time = %g seconds\n", copy_time);
    printf( "n = %d, simulation time = %g seconds\n", n, simulation_time );

    free( particles );
    cudaFree(d_particles);
    cudaFree(particle_list);
    cudaFree(bin_list);
    if( fsave )
        fclose( fsave );

    return 0;
}
