Files = {
    'outrand.txt'
};

Data = CombineResults(Files, false);

plot(Data(:,1), Data(:,2))
xlabel ("Rozmiar kubełka");
ylabel ("Liczba wystąpień");
title ("Rozkład rozmiaru kubełków");
disp(sum(Data(:,2)./1000))