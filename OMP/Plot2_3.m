files_1e6 = readlines('Data_Par/_files_1e6.txt');
files_1e6(cellfun('isempty',files_1e6)) = []; % remove empty cells from mm
% Small = CombineResults(Small, false);
% Med1 = CombineResults(Med1, false);
% Med2 = CombineResults(Med2, false);
Big = CombineResults(files_1e6, false);

% ShowPlot(Small, "mała próba - n = 16");
% ShowPlot(Med1, "średnia próba - n = 1e4");
% ShowPlot(Med2, "duża próba - n = 1e6");
ShowPlot(Big, "n = 5e6");

function ShowPlot(Data, Name)
    hf = figure ();
    hold on 
    Xs = Data(:,1);
    
    plot(Xs, Data(:,2));
    plot(Xs, Data(:,3));
    plot(Xs, Data(:,4));
    plot(Xs, Data(:,5));
    plot(Xs, Data(:,6));
    plot(Xs, Data(:,7));
    xlabel ("Liczba wątków");
    ylabel ("Czas wykonania [ms]");
    title ("Czas wykonania - " +Name);
    legend({'Tworzenie danych', 'Przypisanie do kubełków', 'Łączenie kubełków', 'Sortowanie kubełków', 'Tworzenie wyniku', 'Łącznie'},'Location','northwest')
    hold off

    hf = figure ();
    hold on 
    for i = 2 : width(Data)
        Data(:,i) = Data(1,i) ./ Data(:,i);
    end
    plot(Xs, Data(:,2));
    plot(Xs, Data(:,3));
    plot(Xs, Data(:,4));
    plot(Xs, Data(:,5));
    plot(Xs, Data(:,6));
    plot(Xs, Data(:,7));
    xlabel ("Liczba wątków");
    ylabel ("Przyspieszenie");
    title ("Przyspieszenie - " +Name);
    legend({'Tworzenie danych', 'Przypisanie do kubełków', 'Łączenie kubełków', 'Sortowanie kubełków', 'Tworzenie wyniku', 'Łącznie'},'Location','northwest')
    hold off

    hf = figure ();
    hold on 
    for i = 2 : width(Data)
        Data(:,i) = Data(:,i) ./ Xs;
    end
    plot(Xs, Data(:,2));
    plot(Xs, Data(:,3));
    plot(Xs, Data(:,4));
    plot(Xs, Data(:,5));
    plot(Xs, Data(:,6));
    plot(Xs, Data(:,7));
    xlabel ("Liczba wątków");
    ylabel ("Efektywność");
    title ("Efektywność - " +Name);
    legend({'Tworzenie danych', 'Przypisanie do kubełków', 'Łączenie kubełków', 'Sortowanie kubełków', 'Tworzenie wyniku', 'Łącznie'},'Location','northwest')
    hold off

end