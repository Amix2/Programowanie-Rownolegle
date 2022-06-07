function [OutM] = CombineResults(fileNames, debugPlot)
data = {};
for fileID = 1:length(fileNames)
    fileArr = fileNames(fileID);
    file = fileArr{1,1};
    fileData = readtable(strtrim(file));
    data{fileID} = fileData;
end
dataHeight = height(data{1});
dataWidth = width(data{1});

if(debugPlot)
    hf = figure ();
    hold on 
    for d = 1 : length(data)
        for lineCounter = 1 : height(data{d})
            Ys(lineCounter) = data{d}{lineCounter, 2};
            Xs(lineCounter) = data{d}{lineCounter, 1};
        end
        %plot(Xs, Ys) 
        %scatter(Xs, Ys, 50,'X', 'LineWidth',1);
    end
    legend(fileNames,'Location','southwest')
    title ("DEBUG");
    hold off
end

for columnCounter = 1 : dataWidth
    for lineCounter = 1 : dataHeight
    
        %Xs(lineCounter) = data{1}{lineCounter, 1};
        val = 0;
        count = 0;
        for d = 1 : length(data)
            if lineCounter <= height(data{d})
                val = val + data{d}{lineCounter, columnCounter};
                count = count + 1;
            end
        end
        OutM(lineCounter, columnCounter) = val / count;
    end
end

%OutM(:, 1) = Xs;
%OutM(:, 2) = Ys;
return;
end

