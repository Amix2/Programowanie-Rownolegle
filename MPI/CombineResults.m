function [OutM] = CombineResults(fileNames, debugPlot)
data = {};
for fileID = 1:length(fileNames)
    fileArr = fileNames(fileID);
    file = fileArr{1,1};
    fileData = readtable(file);
    data{fileID} = fileData;
end
dataSize = height(data{1});

if(debugPlot)
    hf = figure ();
    hold on 
    for d = 1 : length(data)
        for counter = 1 : height(data{d})
            Ys(counter) = data{d}{counter, 2};
            Xs(counter) = data{d}{counter, 1};
        end
        plot(Xs, Ys) 
        %scatter(Xs, Ys, 50,'X', 'LineWidth',1);
    end
    legend(fileNames,'Location','southwest')
    title ("DEBUG");
    hold off
end

for counter = 1 : dataSize

    Xs(counter) = data{1}{counter, 1};
    y = 0;
    count = 0;
    for d = 1 : length(data)
        if counter <= height(data{d})
            y = y + data{d}{counter, 2} * 1000;
            count = count + 1;
        end
    end
    Ys(counter) = y / count;
end


OutM(:, 1) = Xs;
OutM(:, 2) = Ys;
return;
end

