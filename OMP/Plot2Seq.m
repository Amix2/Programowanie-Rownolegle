Small = CombineResultsFile('Data_Seq2/_files_1e3.txt', false);
Med1 = CombineResultsFile('Data_Seq2/_files_1e5.txt', false);
Med2 = CombineResultsFile('Data_Seq2/_files_1e6.txt', false);
Big = CombineResultsFile('Data_Seq2/_files_5e6.txt', false);

ShowPlot(Small, "mała próba - n = 1e3");
ShowPlot(Med1, "średnia próba - n = 1e5");
ShowPlot(Med2, "duża próba - n = 1e6");
ShowPlot(Big, "największa próba - n = 5e6");

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
    xlabel ("Liczba kubełków");
    ylabel ("Czas wykonania [ms]");
    title ("Czas wykonania w zależności od liczby kubełków - " +Name);
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
    xlabel ("Liczba kubełków");
    ylabel ("Czas wykonania [ms]");
    title ("Czas wykonania w zależności od liczby kubełków - rozdzielenie na części - " +Name);
    legend({'Tworzenie danych', 'Przypisanie do kubełków', 'Łączenie kubełków', 'Sortowanie kubełków', 'Tworzenie wyniku'},'Location','northwest')
    hold off

end