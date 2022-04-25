function [OutM] = CombineResultsFile(file, debugPlot)
    files_names = readlines(file);
    files_names(cellfun('isempty',files_names)) = [];
    OutM = CombineResults(files_names, debugPlot);
end