Files = {
    'Data2/dataTest.txt'
};

Data = CombineResults(Files, false);

scatter(Data(:,1), Data(:,2), '.')
xlabel ("Indeks kubełka");
ylabel ("Rozmiar kubełka");
title ("Rozkład roamizru kubełków");