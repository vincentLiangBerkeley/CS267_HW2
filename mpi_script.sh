rm mpi.txt
array=(3 6 9 12 18 24)
for j in "${array[@]}"
do 
    echo "Testing with $j nodes."
    srun -N $j -n $j ./mpi -n 50000 -no -s mpi.txt
done


