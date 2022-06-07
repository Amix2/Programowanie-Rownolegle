Data2048 = CombineResultsFile('Results/_files_2048_.txt', false);
hf = figure ();
hold on 
plot(Data2048(:,1), Data2048(:,2), 'b')
plot(Data2048(:,1), Data2048(:,3), 'r')
plot(Data2048(:,1), Data2048(:,4), 'g')
plot(Data2048(:,1), Data2048(:,5), 'm')
xlabel ("Rozmiar bloku");
ylabel ("Czas [ms]");
title ("Czas liczenia transpozycji macierzy 2048x2048 różnymi sposobami");
legend({'Naiwne 1 - discoalesced store', 'Naiwne 2 - discoalesced load', 'Shared 1', 'Shared 2'},'Location','northeast')
hold off

Data8192 = CombineResultsFile('Results/_files_8192_.txt', false);
hf = figure ();
hold on 
plot(Data8192(:,1), Data8192(:,2), 'b')
plot(Data8192(:,1), Data8192(:,3), 'r')
plot(Data8192(:,1), Data8192(:,4), 'g')
plot(Data8192(:,1), Data8192(:,5), 'm')
xlabel ("Rozmiar bloku");
ylabel ("Czas [ms]");
title ("Czas liczenia transpozycji macierzy 8192x8192 różnymi sposobami");
legend({'Naiwne 1 - discoalesced store', 'Naiwne 2 - discoalesced load', 'Shared 1', 'Shared 2'},'Location','northeast')
hold off

Data16384 = CombineResultsFile('Results/_files_16384_.txt', false);
hf = figure ();
hold on 
plot(Data16384(:,1), Data16384(:,2), 'b')
plot(Data16384(:,1), Data16384(:,3), 'r')
plot(Data16384(:,1), Data16384(:,4), 'g')
plot(Data16384(:,1), Data16384(:,5), 'm')
xlabel ("Rozmiar bloku");
ylabel ("Czas [ms]");
title ("Czas liczenia transpozycji macierzy 16384x16384 różnymi sposobami");
legend({'Naiwne 1 - discoalesced store', 'Naiwne 2 - discoalesced load', 'Shared 1', 'Shared 2'},'Location','northeast')
hold off

DataPis = CombineResultsFile('Results/_files_pic_.txt', false);
hf = figure ();
hold on 
plot(DataPis(:,1), DataPis(:,2), 'b')
plot(DataPis(:,1), DataPis(:,3), 'r')
plot(DataPis(:,1), DataPis(:,4), 'g')
xlabel ("Rozmiar bloku");
ylabel ("Czas [ms]");
title ("Czas skalowania obrazu 392:272");
legend({'Skala 0.5', 'Skala 1', 'Skala 2'},'Location','northeast')
ylim([0 0.1])
hold off

hf = figure ();
hold on 
plot(DataPis(:,1), DataPis(:,5), 'b')
plot(DataPis(:,1), DataPis(:,6), 'r')
plot(DataPis(:,1), DataPis(:,7), 'g')
xlabel ("Rozmiar bloku");
ylabel ("Czas [ms]");
title ("Czas skalowania obrazu 623:805");
legend({'Skala 0.5', 'Skala 1', 'Skala 2'},'Location','northeast')
ylim([0 0.2])
hold off
