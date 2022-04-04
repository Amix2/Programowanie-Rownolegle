Single_MPI_Ibsend_Files = {
    'data/Big-Single_MPI_Ibsend_0.txt'
    'data/Big-Single_MPI_Ibsend_1.txt'
    'data/Big-Single_MPI_Ibsend_2.txt'
    'data/Big-Single_MPI_Ibsend_3.txt'
    'data/Big-Single_MPI_Ibsend_4.txt'
    'data/Big-Single_MPI_Ibsend_5.txt'

};

Single_MPI_Ssend_Files = {
    'data/Big-Single_MPI_Ssend_0.txt'
    'data/Big-Single_MPI_Ssend_1.txt'
    'data/Big-Single_MPI_Ssend_2.txt'
    'data/Big-Single_MPI_Ssend_3.txt'
    'data/Big-Single_MPI_Ssend_4.txt'
    };


Single_MPI_Ibsend_M = CombineResults(Single_MPI_Ibsend_Files, true);
Single_MPI_Ssend_M = CombineResults(Single_MPI_Ssend_Files, true);

hf = figure ();

hold on 
scatter(Single_MPI_Ibsend_M(:,1), Single_MPI_Ibsend_M(:,2), '.') 
scatter(Single_MPI_Ssend_M(:,1), Single_MPI_Ssend_M(:,2), '.') 
%set(gca, 'YScale', 'log') 
%axis ([2,max_k, 0, max_time]);
xlabel ("Rozmiar komunikatu [B]");
ylabel ("Przepustowość [Mbit/s]");
title ("Przepustowość dla różnych rozmiarów komunikatów");
legend({'Ibsend', 'Ssend'},'Location','southwest')
hold off
