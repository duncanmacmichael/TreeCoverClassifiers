function [ trainingTime, predictionTime, accuracy, confusionMatrix, ...
    order ] = kFoldCrossValidate_standardClassifiers( k, data, labels, ...
    model )
% kFoldCrossValidate_standardClassifiers
%
% Takes in data, labels, and a desired classifier model and trains that
% model through k-fold cross validation on the dataset. In standard
% classifiers, rows are samples and columns are attributes.
%
%   Input:
%       k: the desired number of folds
%       data: the original data
%       labels: the original class labels
%       model: a string directing the function which model the user would
%           like to train
%   Output:
%       trainingTime: how long it took to train the model
%       predictionTime: how long it took the model to make new predictions
%       accuracy: the accuracy of the trained model
%       confusionMatrix: a confusion matrix for the trained model
%       order: a vector of the order of the observed class labels (for use
%           with the confusion matrix to label what the axes are)

bestTrainingTime = 1000;    % Arbitrarily large number, in seconds
bestPredictionTime = 1000;  % Arbitrarily large number, in seconds
kFoldRunningAccuracy = 0;

%--------------------------------------------------------------------------
% child function for creating an array of matrices, for the confusion
% matrices generated on each pass of the k-fold cross validation
function arrayOfMatrices = createArrayOfMatrices(numArrays, arraySize)
    arrayOfMatrices = cell(1, numArrays);
    for j = 1:numArrays
        arrayOfMatrices{j} = zeros(arraySize);
    end
end
%--------------------------------------------------------------------------
% child function for calculating how many unique class values there are
% in the labels vector, so we can know the size that the confusion matrices
% will be (n unique class labels will result in an nxn confusion matrix)
function numUniqueLabels = findUniqueLabels(labels)
   numUniqueLabels = 0;
   
   % There are seven possible classes, so count occurrences of each in
   % 'labels'
   for j = 1:7
       numOfThisLabel = sum(labels == j);
       if numOfThisLabel > 0
           numUniqueLabels = numUniqueLabels + 1;
       end
   end
end
%--------------------------------------------------------------------------

% create array of matrices to hold each of the eventual confusion matrices
% that are generated from each fold of the k-fold cross validation
confusionMatDimensions = findUniqueLabels(labels);
arrayOfConfusionMats = createArrayOfMatrices(k, confusionMatDimensions);

% Calculate chunk size for k-fold validation and divide inputs into k
% chunks
numSamples = size(data, 1);
chunkSize = floor(numSamples / k);

% Train a classifier with 10-fold cross validation
for i = 1:k
    
    fprintf('fold %d\n', i);  % for debugging to know how far along you are

    % Divide the inputs into 90% training and 10% testing (the 10% is the
    % k-th chunk of this loop)
    offset = (i - 1) * chunkSize + 1;   % Matlab is 1-indexed
    
    trainingChunk1 = data(1 : offset, :);
    trainingChunk2 = data(offset + chunkSize + 1 : end, :);
    currentTraining = vertcat(trainingChunk1, trainingChunk2);

    trainingTargetChunk1 = labels(1 : offset, :);
    trainingTargetChunk2 = labels(offset + chunkSize + 1 : end, :);
    currentTargets = vertcat(trainingTargetChunk1, trainingTargetChunk2);

    testChunk = data(offset : offset + chunkSize, :);
    testTargetChunk = labels(offset : offset + chunkSize, :);

    % Train the model on this fold of the inputs and see how well it
    % did on the training data
    if(strcmp(model, 'naiveBayes'))        
        timeStart = tic;
        trainedModel = fitcnb(currentTraining, currentTargets, ...
            'DistributionNames', 'kernel', 'Width', 0.0072883);
        timeElapsed = toc(timeStart);
    else
        if(strcmp(model, 'decisionTree'))
            timeStart = tic;
            trainedModel = fitctree(currentTraining, currentTargets, ...
                'MinLeaf', 1, 'MaxNumSplits', 5.6657e+05, ...
                'SplitCriterion', 'deviance', 'NumVariablesToSample', 11);
            timeElapsed = toc(timeStart);       
    else
        if(strcmp(model, 'knn'))
            timeStart = tic;
            trainedModel = fitcknn(currentTraining, currentTargets, ...
                'DistanceWeight', 'inverse', 'NumNeighbors', 4);
            timeElapsed = toc(timeStart);
        else % malformed input - must be this format
        fprintf('Model must be "naiveBayes", "decisionTree", or "knn".\n');
        
        % Set unused output variables to nothing to supress Matlab warnings
        trainingTime = []; 
        accuracy = [];
        confusionMatrix = [];
        order = [];
        return;
        end
        end
    end
    
    % Save best training time
    if(timeElapsed < bestTrainingTime)
            bestTrainingTime = timeElapsed;
    end

    % Test the network on the 10% chunk set aside for this fold and measure
    % prediction time
    timeStart = tic;
    predictions = predict(trainedModel, testChunk);
    timeElapsed = toc(timeStart);
    if(timeElapsed < bestPredictionTime)
            bestPredictionTime = timeElapsed;
    end
    
    % Calculate current fold's accuracy and add it to the running total
    numCorrect = sum(predictions == testTargetChunk);
    currentAccuracy = numCorrect / size(testTargetChunk, 1) * 100;
    kFoldRunningAccuracy = kFoldRunningAccuracy + currentAccuracy;   
    
    % add this fold's confusion matrix to the array of all confusion
    % matrices, save the order of classes if it's correct
    [currentConMat, currentOrder]  = confusionmat(testTargetChunk, ...
        predictions); 
    arrayOfConfusionMats{i} = currentConMat;
    order = currentOrder;
end

trainingTime = bestTrainingTime;
predictionTime = bestPredictionTime;
accuracy = kFoldRunningAccuracy / k;

% generate average of confusion matrices
averageConMat = zeros(confusionMatDimensions);
numberOfMatricesUsed = 0;
for m = 1:k
    currentConMatDimensions = size(arrayOfConfusionMats{m});
    
    % check to make sure that fold's confusion matrix has the necessary
    % dimensions. This is because some passes of the k-fold validation
    % apparently don't classify as well and generate confusion matrices
    % of the wrong dimension.
    if(currentConMatDimensions == confusionMatDimensions)
        averageConMat = averageConMat + arrayOfConfusionMats{m};
        numberOfMatricesUsed = numberOfMatricesUsed + 1;
    end
end
averageConMat = ceil(averageConMat ./ numberOfMatricesUsed);
confusionMatrix = averageConMat;

end
