mkdir Results

@echo off
for /L %%n in (1,1,20) do (
    echo Results/2048_%%n.txt
    .\CudaLabCode.exe 1 2048 3000 >> Results/2048_%%n.txt
    echo Results/2048_%%n.txt >> Results/_files_2048_.txt
    sleep 10
)

for /L %%n in (1,1,20) do (
    echo Results/8192_%%n.txt
    .\CudaLabCode.exe 1 8192 300 >> Results/8192_%%n.txt
    echo Results/8192_%%n.txt >> Results/_files_8192_.txt
    sleep 10
)

for /L %%n in (1,1,20) do (
    echo Results/16384_%%n.txt
    .\CudaLabCode.exe 1 16384 30 >> Results/16384_%%n.txt
    echo Results/16384_%%n.txt >> Results/_files_16384_.txt
    sleep 10
)

for /L %%n in (1,1,20) do (
    echo Results/pic_%%n.txt
    .\CudaLabCode.exe 2 >> Results/pic_%%n.txt
    echo Results/pic_%%n.txt >> Results/_files_pic_.txt
    sleep 10
)
