n_particles=(500 1000 2000 4000 8000)
for i in "${n_particles[@]}"
do 
    ./serial -n $i -no -s serial.txt
done

./autograder -v serial -s serial.txt

