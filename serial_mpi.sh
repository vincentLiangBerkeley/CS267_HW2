array=(1000 2000 4000 6000 8000 10000)
for j in "${array[@]}"
do 
    srun -N 4 -n 4 ./mpi -n $j -no -s serial_openmp.txt
done
