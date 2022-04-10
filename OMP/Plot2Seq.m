% Small = {
% 'Data_Seq/1e2_0.txt'
% 'Data_Seq/1e2_1.txt'
% 'Data_Seq/1e2_2.txt'
% 'Data_Seq/1e2_3.txt'
% 'Data_Seq/1e2_4.txt'
% 'Data_Seq/1e2_5.txt'
% };
% 
% Med1 = {
% 'Data_Seq/1e3_0.txt'
% 'Data_Seq/1e3_1.txt'
% 'Data_Seq/1e3_2.txt'
% 'Data_Seq/1e3_3.txt'
% 'Data_Seq/1e3_4.txt'
% 'Data_Seq/1e3_5.txt'
% };
% 
% Med2 = {
% 'Data_Seq/1e5_0.txt'
% 'Data_Seq/1e5_1.txt'
% 'Data_Seq/1e5_2.txt'
% 'Data_Seq/1e5_3.txt'
% 'Data_Seq/1e5_4.txt'
% 'Data_Seq/1e5_5.txt'
% };
% 
% Big = {
% 'Data_Seq/1e6_0.txt'
% 'Data_Seq/1e6_1.txt'
% 'Data_Seq/1e6_2.txt'
% 'Data_Seq/1e6_3.txt'
% };

Small = {
'Data_Seq2/1e2_0.txt'
'Data_Seq2/1e2_1.txt'
'Data_Seq2/1e2_2.txt'
'Data_Seq2/1e2_3.txt'
'Data_Seq2/1e2_4.txt'
};

Med1 = {
'Data_Seq2/1e3_0.txt'
'Data_Seq2/1e3_1.txt'
'Data_Seq2/1e3_2.txt'
'Data_Seq2/1e3_3.txt'
'Data_Seq2/1e3_4.txt'
};

Med2 = {
'Data_Seq2/1e5_0.txt'
'Data_Seq2/1e5_1.txt'
'Data_Seq2/1e5_2.txt'
'Data_Seq2/1e5_3.txt'
'Data_Seq2/1e5_4.txt'
};

Big = {
'Data_Seq2/1e6_0.txt'
'Data_Seq2/1e6_1.txt'
'Data_Seq2/1e6_2.txt'
'Data_Seq2/1e6_3.txt'
'Data_Seq2/1e6_4.txt'
};

Small = CombineResults(Small, false);
Med1 = CombineResults(Med1, false);
Med2 = CombineResults(Med2, false);
Big = CombineResults(Big, false);

ShowPlot(Small, "mała próba - n = 1e2");
ShowPlot(Med1, "średnia próba - n = 1e3");
ShowPlot(Med2, "duża próba - n = 1e5");
ShowPlot(Big, "największa próba - n = 1e6");

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