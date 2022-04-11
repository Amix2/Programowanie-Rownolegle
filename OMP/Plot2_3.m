% Small = {
%     'Data/small_0.txt'
%     'Data/small_1.txt'
%     'Data/small_2.txt'
%     %'Data/small_3.txt'
%     'Data/small_4.txt'
%     'Data/small_5.txt'
%     'Data/small_6.txt'
%     'Data/small_7.txt'
%     'Data/small_8.txt'
%     'Data/small_9.txt'
%     'Data/small_10.txt'
% };
% 
% Med1 = {
%     'Data/med1_0.txt'
%     'Data/med1_1.txt'
%     'Data/med1_2.txt'
%     'Data/med1_3.txt'
%     'Data/med1_4.txt'
%     %'Data/med1_5.txt'
%     'Data/med1_6.txt'
%     'Data/med1_7.txt'
%     'Data/med1_8.txt'
%     'Data/med1_9.txt'
%     'Data/med1_10.txt'
% };
% 
% Med2 = {
%     'Data/med2_0.txt'
%     %'Data/med2_1.txt'
%     'Data/med2_2.txt'
%     %'Data/med2_3.txt'
%     'Data/med2_4.txt'
%     'Data/med2_5.txt'
%     'Data/med2_6.txt'
%     'Data/med2_7.txt'
%     %'Data/med2_8.txt'
%     'Data/med2_9.txt'
%  'Data/med2_10.txt'
% };
% 
% Big = {
%     'Data_Par/1e6_1.txt'
%     'Data_Par/1e6_2.txt'
%     'Data_Par/1e6_3.txt'
%     'Data_Par/1e6_4.txt'
%     'Data_Par/1e6_5.txt'
%     'Data_Par/1e6_6.txt'
%     'Data_Par/1e6_7.txt'
%     'Data_Par/1e6_8.txt'
%     'Data_Par/1e6_9.txt'
%     'Data_Par/1e6_10.txt'
% };
files_1e6 = readlines('Data_Par/_files_1e6.txt');
files_1e6(cellfun('isempty',files_1e6)) = []; % remove empty cells from mm
% Small = CombineResults(Small, false);
% Med1 = CombineResults(Med1, false);
% Med2 = CombineResults(Med2, false);
Big = CombineResults(files_1e6, false);

% ShowPlot(Small, "mała próba - n = 16");
% ShowPlot(Med1, "średnia próba - n = 1e4");
% ShowPlot(Med2, "duża próba - n = 1e6");
ShowPlot(Big, "n = 1e6");

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