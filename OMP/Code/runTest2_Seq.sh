g++ lab4.cpp -o lab4_o -fopenmp -std=c++11
mkdir Data_Seq2
for C in {0..4}
do
    echo $C
    ./lab4_o 2 100 >> Data_Seq2/1e2_$C.txt
    ./lab4_o 2 1000 >> Data_Seq2/1e3_$C.txt
    ./lab4_o 2 100000 >> Data_Seq2/1e5_$C.txt
    ./lab4_o 2 1000000 >> Data_Seq2/1e6_$C.txt
done
