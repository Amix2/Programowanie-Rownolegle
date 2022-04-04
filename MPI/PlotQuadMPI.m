Quad_MPI_Ibsend_Files = {
    'Quad_MPI_Ibsend_0.txt'
    'Quad_MPI_Ibsend_1.txt'
    'Quad_MPI_Ibsend_2.txt'
    'Quad_MPI_Ibsend_3.txt'
    'Quad_MPI_Ibsend_4.txt'
    'Quad_MPI_Ibsend_5.txt'
};

Quad_MPI_Ssend_Files = {
    'Quad_MPI_Ssend_0.txt'
    'Quad_MPI_Ssend_1.txt'
    'Quad_MPI_Ssend_2.txt'
    'Quad_MPI_Ssend_3.txt'
    'Quad_MPI_Ssend_4.txt'
    'Quad_MPI_Ssend_5.txt'
};


Quad_MPI_Ibsend_M = CombineResults(Quad_MPI_Ibsend_Files, true);
Quad_MPI_Ssend_M = CombineResults(Quad_MPI_Ssend_Files, true);

hf = figure ();
hold on 
plot(Quad_MPI_Ibsend_M(:,1), Quad_MPI_Ibsend_M(:,2)) 
plot(Quad_MPI_Ssend_M(:,1), Quad_MPI_Ssend_M(:,2)) 
%set(gca, 'YScale', 'log') 
%axis ([2,max_k, 0, max_time]);
xlabel ("Rozmiar komunikatu [B]");
ylabel ("Przepustowość [Mbit/s]");
title ("Przepustowość dla różnych roamiarów komunikatów");
legend({'Dzielona maszyna Ibsend', 'Dzielona maszyna Ssend'},'Location','southwest')
hold off
