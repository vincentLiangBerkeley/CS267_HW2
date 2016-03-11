#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "common.h"
#include "bin.h"

#define DEBUG 0
#define A_FLAG 0
#define RANK 0
#define D_FLAG DEBUG && rank == RANK
#define DIETAG 1000000
#define CHECK_GHOST 0
#define GHOST 2
//
//  benchmarking program
//
//
bool isPrime(int n)
{
    if (n <= 2) return true;
    for(int i = 2; i < sqrt(n); i ++)
        if (n % i == 0) return false;
    return true;
}

int main( int argc, char **argv )
{    
    int navg, nabsavg=0;
    double dmin, absmin=1.0,davg,absavg=0.0;
    double rdavg,rdmin;
    int rnavg; 
 
    //
    //  process command line parameters
    //
    if( find_option( argc, argv, "-h" ) >= 0 )
    {
        printf( "Options:\n" );
        printf( "-h to see this help\n" );
        printf( "-n <int> to set the number of particles\n" );
        printf( "-o <filename> to specify the output file name\n" );
        printf( "-s <filename> to specify a summary file name\n" );
        printf( "-no turns off all correctness checks and particle output\n");
        return 0;
    }
    
    int n = read_int( argc, argv, "-n", 1000 );
    char *savename = read_string( argc, argv, "-o", NULL );
    char *sumname = read_string( argc, argv, "-s", NULL );
    
    //
    //  set up MPI
    //
    int n_proc, rank;
    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &n_proc );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    
    //
    //  allocate generic resources
    //
    FILE *fsave = savename && rank == 0 ? fopen( savename, "w" ) : NULL;
    FILE *fsum = sumname && rank == 0 ? fopen ( sumname, "a" ) : NULL;


    particle_t *particles = (particle_t*) malloc( n * sizeof(particle_t) );
    
    MPI_Datatype PARTICLE;
    MPI_Type_contiguous( 6, MPI_DOUBLE, &PARTICLE );
    MPI_Type_commit( &PARTICLE );
        
    //
    //  initialize and distribute the particles (that's fine to leave it unoptimized)
    //
    double grid_size = set_size( n );

    if( rank == 0 )
        init_particles( n, particles );
    MPI_Bcast(particles, n, PARTICLE, 0, MPI_COMM_WORLD);

    // Set up bin sizes
    int bin_i, bin_j, num_bins = n % 4 == 0 ? n/4:n/4+1;
    bin_t *bin_list = (bin_t*) malloc(num_bins * sizeof(bin_t));
    if (D_FLAG) printf("Testing initializing bins: \n");
    set_grid_size(bin_i, bin_j, num_bins);
    if (D_FLAG) printf("There are %d bins, %d per row with %d rows.\n", num_bins, bin_i, bin_j);
    double bin_x = grid_size / bin_i, bin_y = grid_size / bin_j;
    if (D_FLAG) printf("The bins are of size %f by %f, err = %f\n", bin_y, bin_x, bin_x*bin_y*num_bins - grid_size*grid_size);
    init_grid(num_bins, bin_list);
    bin_particles(n, particles, num_bins, bin_list, bin_x, bin_y, bin_j);


    // Setting up bin partitioning across the cluster, this is 1D layout
    int num_stripes = num_bins / min(bin_i, bin_j);
    int rows_per_proc = num_stripes / n_proc; // This assumes n_proc is not too large
    int local_start = rank*rows_per_proc, local_stripes = rank == n_proc-1 ? num_stripes-rows_per_proc*rank : rows_per_proc;


    int row_start = 0, row_end = bin_j,
        col_start = local_start, col_end = local_start + local_stripes;
 
    if (A_FLAG)
    {
        printf("Testing bin allocation across cluster:\n");
        printf("num_stripes = %d, rows_per_proc = %d, node %d starts at %d and has %d stripes.\n", num_stripes, rows_per_proc, rank, local_start, local_stripes);
        int sum = 0;
        for (int i = 0; i < n; i ++) 
            if (particles[i].vx != -1) sum++;
        printf("Node %d has %d particles.\n", rank, sum);
    }


    //
    //  simulate a number of time steps
    //
    double simulation_time = read_timer( );
    MPI_Request send_request;
    MPI_Status status;
    for( int step = 0; step < 1000; step++ )
    {
        navg = 0;
        dmin = 1.0;
        davg = 0.0;
        
        //
        //  save current step if necessary (slightly different semantics than in other codes)
        //
        if( find_option( argc, argv, "-no" ) == -1 )
          if( fsave && (step%SAVEFREQ) == 0 )
            save( fsave, n, particles );
        
        //
        //  compute all forces, only to particles in the region for this node
        //
        int locals[n];
        int local_size = 0;
        for (int c = col_start; c < col_end; c ++)
        {
            for(int r = row_start; r < row_end; r ++)
            {
                bin_t this_bin = bin_list[r + c*bin_j];
                for(int p = 0; p < this_bin.bin_size; p ++)
                {
                    int i = this_bin.indeces[p];
                    locals[local_size ++] = i;

                    particles[i].ax = particles[i].ay = 0;
                    for(int c_n = max(c-1, 0); c_n <= min(c+1,bin_i-1); c_n++)
                    {
                        for(int r_n = max(r-1,0); r_n <= min(r+1, bin_j-1); r_n++)
                        {
                            bin_t neighbor = bin_list[r_n + c_n*bin_j];
                            //printf("Neighbor index = %d with size: %d\n", r+c*bin_j, neighbor.bin_size);
                            for(int j = 0; j < neighbor.bin_size; j ++)
                                apply_force(particles[i], particles[neighbor.indeces[j]], &dmin, &davg, &navg);   
                        }
                    }
                }
            }
        }
        
        // Clear the ghost zones
        if (col_start > 0) clear_bin_col(bin_list, row_start, row_end, col_start - 1, bin_j);
        if (col_end < bin_i) clear_bin_col(bin_list, row_start, row_end, col_end, bin_j);

        // Move the particles, send particles to destination nodes if needed
        for(int p = 0; p < local_size; p ++)
        {
            int i = locals[p];
            int r_old = particles[i].y / bin_y, c_old = particles[i].x / bin_x;
            move( particles[i] );
            if (abs(particles[i].vx) > 5 || abs(particles[i].vy) > 5) 
                printf("A particle in (%d, %d) is moving at (%f, %f)\n", r_old, c_old, particles[i].vx, particles[i].vy);
            int r = particles[i].y / bin_y, c = particles[i].x / bin_x;

            if (r != r_old || c != c_old)
            {
                remove_particle(bin_list, i, r_old + c_old*bin_j);
                add_particle(bin_list, i, r + c*bin_j);
            }

            if (c < col_start || c >= col_end)
            {
                int target = c < col_start ? rank - 1:rank + 1;
                int new_node = min(c / rows_per_proc, n_proc - 1);
                if (new_node != target) printf("There's a particle moved from node %d to node %d.\n", rank, new_node);
                MPI_Request request;
                MPI_Isend(&particles[i], 1, PARTICLE, target, i, MPI_COMM_WORLD, &request);
            }
        }

        if (D_FLAG) printf("Node %d has finished applying force and moving particles.\n", rank);

        MPI_Request send_request;
        // Finished moving and sending particles, now send ghost zones:
        for(int r = row_start; r < row_end; r++)
        {
            // if(D_FLAG) printf("Node %d is sending particles in row %d.\n", rank, r);
            if(rank != 0)
            {
                bin_t start_bin = bin_list[r + col_start*bin_j];
                for(int p = 0; p < start_bin.bin_size; p ++)
                {
                    int i = start_bin.indeces[p];
                    if (DEBUG && rank == n_proc - 1) printf("Node %d is sending bin (%d, %d) to node %d\n", rank, r, col_start, rank - 1);
                    MPI_Isend(&particles[i], 1, PARTICLE, rank - 1, i, MPI_COMM_WORLD, &send_request);
                }
            }

            if (rank != n_proc - 1)
            {
                bin_t end_bin = bin_list[r + (col_end - 1)*bin_j];
                for(int p = 0; p < end_bin.bin_size; p ++)
                {
                    int i = end_bin.indeces[p];
                    // if (DEBUG && rank == n_proc - 2) printf("Node %d is sending particle %d to node %d\n", rank, i, rank + 1);
                    MPI_Isend(&particles[i], 1, PARTICLE, rank + 1, i , MPI_COMM_WORLD, &send_request);
                }
            }
        }

        // Now send a terminating message to neighbors
        if (rank != 0)
            MPI_Isend(0, 0, MPI_INT, rank - 1, DIETAG, MPI_COMM_WORLD, &send_request);
        if (rank != n_proc - 1)
            MPI_Isend(0, 0, MPI_INT, rank + 1, DIETAG, MPI_COMM_WORLD, &send_request);

        particle_t temp;
        int term_count = 0;
        int term_limit = (rank == 0 || rank == n_proc - 1) ? 1 : 2;

        while(term_count < term_limit && n_proc > 1)
        {
            MPI_Recv(&temp, 1, PARTICLE, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            int i = status.MPI_TAG;
            if (i == DIETAG ) 
            {
                term_count ++;
                continue;
            }
            particles[i].x = temp.x;
            particles[i].y = temp.y;
            particles[i].vx = temp.vx;
            particles[i].vy = temp.vy;
            particles[i].ax = temp.ax;
            particles[i].ay = temp.ay;
            
            int r = particles[i].y/bin_y, c = particles[i].x/bin_x;
            add_particle(bin_list, i, r + c*bin_j);
        }

        
        // Check ghost zones explicitly, only the last one
        if (CHECK_GHOST)
        {
            if (rank == GHOST)
            {
                printf("Iteration %d\n\n", step);
                printf("The right ghost zone for Node %d is:\n", GHOST);
                for(int r = row_start; r < row_end; r ++)
                {
                    bin_t this_bin = bin_list[r + (col_end)*bin_j];
                    for(int p = 0; p < this_bin.bin_size; p ++)
                        printf("%d ", this_bin.indeces[p]);
                    printf("\n");
                }
            }

            if (rank == GHOST + 1)
            {
                
                printf("The left ghost zone for Node %d is\n", GHOST + 1);
                for(int r = row_start; r < row_end; r ++)
                {
                    bin_t this_bin = bin_list[r + (col_start)*bin_j];
                    for(int p = 0; p < this_bin.bin_size; p ++)
                        printf("%d ", this_bin.indeces[p]);
                    printf("\n");
                }
            }   
        }
        

        if( find_option( argc, argv, "-no" ) == -1 )
        {
          
          MPI_Reduce(&davg,&rdavg,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&navg,&rnavg,1,MPI_INT,MPI_SUM,0,MPI_COMM_WORLD);
          MPI_Reduce(&dmin,&rdmin,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);

 
          if (rank == 0){
            //
            // Computing statistical data
            //
            if (rnavg) {
              absavg +=  rdavg/rnavg;
              nabsavg++;
            }
            if (rdmin < absmin) absmin = rdmin;
          }
        }
        MPI_Barrier(MPI_COMM_WORLD);    
    }
    simulation_time = read_timer( ) - simulation_time;
  
    if (rank == 0) {  
      printf( "n = %d, simulation time = %g seconds", n, simulation_time);

      if( find_option( argc, argv, "-no" ) == -1 )
      {
        if (nabsavg) absavg /= nabsavg;
      // 
      //  -The minimum distance absmin between 2 particles during the run of the simulation
      //  -A Correct simulation will have particles stay at greater than 0.4 (of cutoff) with typical values between .7-.8
      //  -A simulation where particles don't interact correctly will be less than 0.4 (of cutoff) with typical values between .01-.05
      //
      //  -The average distance absavg is ~.95 when most particles are interacting correctly and ~.66 when no particles are interacting
      //
      printf( ", absmin = %lf, absavg = %lf", absmin, absavg);
      if (absmin < 0.4) printf ("\nThe minimum distance is below 0.4 meaning that some particle is not interacting");
      if (absavg < 0.8) printf ("\nThe average distance is below 0.8 meaning that most particles are not interacting");
      }
      printf("\n");     
        
      //  
      // Printing summary data
      //  
      if( fsum)
        fprintf(fsum,"%d %d %g\n",n,n_proc,simulation_time);
    }
  
    //
    //  release resources
    //
    if ( fsum )
        fclose( fsum );
    clear_grid(num_bins, bin_list);
    free(bin_list);
    free( particles );
    if( fsave )
        fclose( fsave );
    
    MPI_Finalize( );
    
    return 0;
}
