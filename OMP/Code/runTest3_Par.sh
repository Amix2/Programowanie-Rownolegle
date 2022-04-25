g++ lab4_win.cpp -o lab4_o -fopenmp -std=c++11 -O2
mkdir Data_Par
for C in {0..20}
do
    echo $C
    echo 'Data_Par/1e6_$C'
    ./lab4_o 3 10000000 >> Data_Par/1e6_$C.txt
    echo Data_Par/1e6_$C.txt >> Data_Par/_files_1e6.txt
)
