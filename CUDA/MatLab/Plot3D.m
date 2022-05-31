data = readmatrix("data.txt");

X = data(:,1);
Y = data(:,2);
Z = data(:,3);

[A,B,C] = CreateMesh(X, Y, Z);

hf = figure ();
plot(B, C(:,100))
xlabel ("Liczba wątków w bloku");
ylabel ("Czas wykonania [ms]");

hf = figure ();
mesh(A,B,C)
ylabel ("Liczba wątków w bloku");
xlabel ("Rozmiar wektora");
zlabel ("Czas wykonania [ms]");


function [X, Y, Z] = CreateMesh(iX, iY, iZ)
    X = unique(iX);
    Y = unique(iY);
    for i = 1:length(iX)
        xPos = find(X == iX(i));
        yPos = find(Y == iY(i));
        Z(yPos, xPos) = min(iZ(i), 5);
    end
end