rm openmp.txt
n_threads=(1 2 4 6 12 18 24)
n_particles=(1000 2000 4000 3000 6000 9000 12000)
for j in "${n_threads[@]}"
do 
    export OMP_NUM_THREADS=$j
    ./openmp -n 500 -no -s openmp.txt
done 

for i in {0..6}
do 
    export OMP_NUM_THREADS=${n_threads[$i]}
    ./openmp -n ${n_particles[$i]} -no -s openmp.txt
done

./autograder -v openmp -s openmp.txt
