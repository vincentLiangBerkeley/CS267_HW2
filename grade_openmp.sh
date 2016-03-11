rm openmp.txt
n_threads=(2 4 6 12 18 24)
n_particles=(1000 2000 4000 3000 6000 9000 12000)
for j in "${n_threads[@]}"
do 
    export OMP_NUM_THREADS=$j
    srun -N 1 -n 1 -c $j ./openmp -n 50000 -no -s openmp.txt
done
