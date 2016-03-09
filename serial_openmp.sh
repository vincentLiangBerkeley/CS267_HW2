array=(1000 2000 4000 6000 8000 10000)
for j in "${array[@]}"
do 
    srun -n 1 -c 2 ./openmp -n $j -no -s openmp.txt
done
