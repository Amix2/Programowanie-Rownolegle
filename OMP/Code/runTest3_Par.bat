g++ lab4.cpp -o lab4_o -fopenmp -std=c++11
mkdir Data_Par
@echo off
for /L %%n in (1,1,100) do (
    echo 'Data_Par/1e6_%%n'
    .\lab4_o.exe 3 1000000 >> Data_Par/1e6_%%n.txt
    echo Data_Par/1e6_%%n.txt >> Data_Par/_files_1e6.txt
)
