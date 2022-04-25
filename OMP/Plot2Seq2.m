
files_1e6 = readlines('Data_Seq3/_files_5e6.txt');
files_1e6(cellfun('isempty',files_1e6)) = []; % remove empty cells from mm
%Small = CombineResults(Small, false);
%Med1 = CombineResults(Med1, false);
%Med2 = CombineResults(Med2, false);
Big = CombineResults(files_1e6, false);

%ShowPlot(Small, "mała próba - n = 1e2");
%ShowPlot(Med1, "średnia próba - n = 1e3");
%ShowPlot(Med2, "duża próba - n = 1e5");
ShowPlot(Big, "t=1 n=5e6");

function ShowPlot(Data, Name)
    hf = figure ();
    hold on 
    Xs = Data(:,1);
    
    plot(Xs, Data(:,7));
%     plot(Xs, Data(:,3));
%     plot(Xs, Data(:,4));
%     plot(Xs, Data(:,5));
%     plot(Xs, Data(:,6));
%     plot(Xs, Data(:,7));
%     plot(Xs, Data(:,8));
    xlabel ("Wielkość tablicy danych n");
    ylabel ("Czas wykonania [ms]");
    title ("Czas wykonania w zależności od rozmiary tablicy - " +Name);
    legend({'Sortowanie'},'Location','southeast')
    hold off

    hf = figure ();
    hold on 
    Xs = Data(:,1);
    
    plot(Xs, Data(:,2));
    plot(Xs, Data(:,3));
    plot(Xs, Data(:,4));
    plot(Xs, Data(:,5));
    plot(Xs, Data(:,6));
    xlabel ("Wielkość tablicy danych n");
    ylabel ("Czas wykonania [ms]");
    title ("Czas wykonania w zależności od rozmiary tablicy - " +Name);
    legend({'Tworzenie danych', 'Przypisanie do kubełków', 'Łączenie kubełków', 'Sortowanie kubełków', 'Tworzenie wyniku'},'Location','northwest')
    hold off

end