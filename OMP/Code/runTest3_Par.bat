g++ lab4_win.cpp -o lab4_o -fopenmp -std=c++11 -O2
mkdir Data_Seq4
@echo off
for /L %%n in (1,1,20) do (
    echo 'Data_Seq4/1e6_%%n'
    .\lab4_o.exe 2 1000000 >> Data_Seq4/5e6_%%n.txt
    echo Data_Seq4/5e6_%%n.txt >> Data_Seq4/_files_5e6.txt
)
