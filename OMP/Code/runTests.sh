gcc -Wall lab3.c -o lab3_o -fopenmp
for C in {0..10}
do
echo $C
#echo "small"
#./lab3_o 16 4 >> Data/small_$C.txt
#echo "med1"
#./lab3_o 10000 4 >> Data/med1_$C.txt
echo "med2"
./lab3_o 1000000 4 >> Data/med2_$C.txt
#echo "big"
#./lab3_o 100000000 4 >> Data/big_$C.txt
done
