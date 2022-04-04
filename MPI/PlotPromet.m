Files_Small = {
'data/small_1_5.txt'
'data/small_2_1.txt'
'data/small_2_2.txt'
'data/small_2_3.txt'
'data/small_2_4.txt'
'data/small_2_5.txt'
%'data/small_3_1.txt' %1
%'data/small_3_2.txt' %1
%'data/small_3_3.txt' %1
%'data/small_3_4.txt'
%'data/small_3_5.txt' %1
'data/small_4_1.txt'
'data/small_4_2.txt'
'data/small_4_3.txt'
'data/small_4_4.txt'
'data/small_4_5.txt'
'data/small_5_1.txt'
'data/small_5_2.txt'
'data/small_5_3.txt'
'data/small_5_4.txt'
'data/small_5_5.txt'
%'data/small_6_1.txt' %1
%'data/small_6_2.txt' %1
%'data/small_6_3.txt' %1
%'data/small_6_4.txt' %1
%'data/small_6_5.txt'
};
Files_Med = {
    %'data/test_10000000000_0.txt'
    'data/test_10000000000_00.txt'
    'data/test_10000000000_1.txt'
    'data/test_10000000000_2.txt'
   % 'data/test_10000000000_3.txt'
    %'data/7-test-10000000000.txt'
    'data/8-test-10000000000.txt'
    'data/9-test-10000000000.txt'
    %'data/11-test-10000000000.txt'
    'data/15-test-10000000000.txt'
    'data/med_1'
%'data/med_2' %
'data/med_3'
%'data/med_4'%
'data/med_5'
'data/med_6'
%'data/med_7'%
'data/med_8'
%'data/med_9'
%'data/med_10'%
'data/med_11'
'data/med_12'

};

Files_Big = {
    'data/test_20000000000_0.txt'
    'data/test_20000000000_00.txt'
    'data/test_20000000000_1.txt'
    'data/test_20000000000_2.txt'
    'data/test_20000000000_3.txt'
    %'data/7-test-20000000000.txt'
    'data/8-test-20000000000.txt'
    'data/9-test-20000000000.txt'
    %'data/11-test-20000000000.txt'
    'data/15-test-20000000000.txt'
    'data/big_1'
%'data/big_2'%
'data/big_3'
%'data/big_4'%
'data/big_5'
'data/big_6'
%'data/big_7'%
'data/big_8'
'data/big_9'
%'data/big_10'
'data/big_11'
'data/big_12'


};


Files_Small_M = CombineResults(Files_Small, true);
Files_Med_M = CombineResults(Files_Med, true);
Files_Big_M = CombineResults(Files_Big, true);

smallProblemTime = 3.562730267; %Files_Small_M(1,2) * 0.99;
medProblemTime = 355.8374301;%Files_Med_M(1,2) * 0.99;
bigProblemTime = 703.9505788;%Files_Big_M(1,2);

smallProblemTime = Files_Small_M(1,2);
medProblemTime = Files_Med_M(1,2);
bigProblemTime = Files_Big_M(1,2);

hf = figure ();
threadsXs = Files_Small_M(:,1);
smallTimeYs = Files_Small_M(:,2);
medTimeYs = Files_Med_M(:,2);
bigTimeYs = Files_Big_M(:,2);


for i = 1:length(Files_Small_M(:,1))
    smallIdealYs(i) = smallProblemTime / Files_Small_M(i,1);
end



hf = figure ();
hold on 
scatter(threadsXs, smallTimeYs, 'filled') 
scatter(threadsXs, smallIdealYs,50,'X', 'LineWidth',1);
xlabel ("Liczba wątków");
ylabel ("Czas wykonania zadania [s]");
title ("Czas wykonania małego zadania");
legend({'Mały problem', 'Idealny przebieg'},'Location','southwest')
hold off


for i = 1:length(Files_Med_M(:,1))
    medIdealYs(i) = medProblemTime / Files_Med_M(i,1);
end

hf = figure ();
hold on 
scatter(threadsXs, medTimeYs, 'filled') 
scatter(threadsXs, medIdealYs,50,'X', 'LineWidth',1);
xlabel ("Liczba wątków");
ylabel ("Czas wykonania zadania [s]");
title ("Czas wykonania średniego zadania");
legend({'Średni problem', 'Idealny przebieg'},'Location','southwest')
hold off


for i = 1:length(Files_Big_M(:,1))
    bigIdealYs(i) = bigProblemTime / Files_Big_M(i,1);
end

hf = figure ();
hold on 
scatter(threadsXs, bigTimeYs, 'filled') 
scatter(threadsXs, bigIdealYs,50,'X', 'LineWidth',1);
xlabel ("Liczba wątków");
ylabel ("Czas wykonania zadania [s]");
title ("Czas wykonania dużego zadania");
legend({'Duży problem', 'Idealny przebieg'},'Location','southwest')
hold off

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idealSpeedupYs = 1:12;
for i = 1:length(threadsXs)
    smallSpeedupYs(i) = smallProblemTime / smallTimeYs(i);
    medSpeedupYs(i) = medProblemTime / medTimeYs(i);
    bigSpeedupYs(i) = bigProblemTime / bigTimeYs(i);
end

hf = figure ();
hold on 
scatter(threadsXs, smallSpeedupYs, 'filled') 
scatter(threadsXs, medSpeedupYs, 'filled') 
scatter(threadsXs, bigSpeedupYs, 'filled') 
scatter(threadsXs, idealSpeedupYs,50,'X', 'LineWidth',1);
xlabel ("Liczba wątków");
ylabel ("Przyspieszenie");
title ("Przyspieszenie");
legend({'Mały problem', 'Średni problem','Duzy problem', 'Idealny przebieg'},'Location','southwest')
hold off

for i = 1:length(threadsXs)
    idealEffiYs(i) = 1;
    smallEffiYs(i) = smallSpeedupYs(i) / threadsXs(i);
    medEffiYs(i) = medSpeedupYs(i) / threadsXs(i);
    bigEffiYs(i) = bigSpeedupYs(i) / threadsXs(i);
end

hf = figure ();
hold on 
scatter(threadsXs, smallEffiYs, 'filled') 
scatter(threadsXs, medEffiYs, 'filled') 
scatter(threadsXs, bigEffiYs, 'filled') 
scatter(threadsXs, idealEffiYs,50,'X', 'LineWidth',1);
xlabel ("Liczba wątków");
ylabel ("Efektywność");
title ("Efektywność");
legend({'Mały problem', 'Średni problem','Duzy problem','Idealny przebieg'},'Location','southwest')
hold off

for i = 1:length(threadsXs)
    smallSerialFracXs(i) = (1/smallSpeedupYs(i) - 1 / threadsXs(i)) / (1 - 1 / threadsXs(i));
    medSerialFracXs(i) = (1/medSpeedupYs(i) - 1 / threadsXs(i)) / (1 - 1 / threadsXs(i));
    bigSerialFracXs(i) = (1/bigSpeedupYs(i) - 1 / threadsXs(i)) / (1 - 1 / threadsXs(i));
end

hf = figure ();
hold on 
scatter(threadsXs, smallSerialFracXs, 'filled') 
scatter(threadsXs, medSerialFracXs, 'filled') 
scatter(threadsXs, bigSerialFracXs, 'filled') 
%scatter(threadsXs, idealEffiYs,50,'X', 'LineWidth',1);
xlabel ("Liczba wątków");
ylabel ("Część sekwencyjna");
title ("Zależność części sekwencyjnej od liczby procesorów ");
legend({'Mały problem','Średni problem','Duzy problem'},'Location','southwest')
hold off


% hf = figure ();
% hold on 
% scatter(Files_Med_M(:,1), Files_Med_M(:,2), 'filled') 
% problemTime = Files_Med_M(1,2);
% for i = 1:length(Files_Med_M(:,1))
%     idealYs(i) = problemTime / Files_Med_M(i,1)
% end
% scatter(Files_Med_M(:,1), idealYs, '+')
% xlabel ("Liczba wątków");
% ylabel ("Czas wykonania zadania [s]");
% title ("Czas wykonania średniego zadania");
% legend({'Średni problem', 'Idealny przebieg'},'Location','southwest')
% hold off
