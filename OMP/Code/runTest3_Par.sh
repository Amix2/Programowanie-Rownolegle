g++ lab4.cpp -o lab4_o -fopenmp -std=c++11
mkdir Data_Par
for C in {0..10}
do
    echo $C
    ./lab4_o 3 100 >> Data_Par/1e2_$C.txt
    ./lab4_o 3 1000 >> Data_Par/1e3_$C.txt
    ./lab4_o 3 100000 >> Data_Par/1e5_$C.txt
    ./lab4_o 3 1000000 >> Data_Par/1e6_$C.txt
done
