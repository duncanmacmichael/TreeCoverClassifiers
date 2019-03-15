function [ preProcessedData, labels ] = preProcessData( filename, ...
    numPrincipalComponents )
% preProcessData 
%
% Takes in original data file, collapses binary columns, and
% normalizes the data. Assumes first row in source is the attribute labels
% and skips that row.
%
%   Input:
%       filename: the CSV file being read in. In this project, columns 1-10
%           are data, 11-14 are binary columns for the wilderness type,
%           15-54 are binary columns for soil type, and 55 is the class
%           label.
%       numPrincipalComponents: however many principal components are
%           desired for Principal Component Analysis on the data. The
%           function will eliminate all columns that are not in this top
%           number of principal components. If input is out of range (must
%           be > 0 and <= 11) then the function skips PCA, giving
%           flexibility to the user to determine if they want to try
%           without PCA at all.
%   Output:
%       preProcessedData: the processed data matrix
%       labels: the class labels

% Read the data. No error checking for now.
originalData = csvread(filename, 1);
columnHeight = size(originalData, 1);
labels = originalData(:, 55);

preProcessedData = originalData(1 : end, 1 : 10);

% Collapse the 40 soil type columns into one column for the processed data.
% Map columns 15-54 (originally binary values) to a 1-40 system for new
% single column
soilTypeColumn = zeros(columnHeight, 1);
for rowIndex = 1:columnHeight
    for soilTypeColumnIndex = 15:54
        if(originalData(rowIndex, soilTypeColumnIndex) == 1)
            soilTypeColumn(rowIndex) = soilTypeColumnIndex - 14;
            break;
        end
    end                
end
preProcessedData = horzcat(preProcessedData, soilTypeColumn);

% Apply PCA to data and eliminate columns outside the desired top PCA range
% (in other words, if 4 principal components are desired, only return those
% top four columns)

if(numPrincipalComponents > 0 && numPrincipalComponents <= 11)
    pcaData = pca(preProcessedData);
    principalComponents = pcaData(:, 1 : numPrincipalComponents);
    preProcessedData = preProcessedData * principalComponents;
else
    fprintf('NumPrincipalComponents must be > 0 and <= 11. ... Skipping PCA.\n');
end

% Normalize all data into the 0-1 range per column
rowWidth = size(preProcessedData, 2);
normalizedData = zeros(columnHeight, rowWidth);
for i = 1:rowWidth
    currentColumn = preProcessedData(:, i);
    columnMax = max(currentColumn);
    columnMin = min(currentColumn);
    for j = 1:columnHeight
        normalizedData(j, i) = ((preProcessedData(j, i) - columnMin) / ...
            (columnMax - columnMin));
    end
end

preProcessedData = normalizedData;

end

