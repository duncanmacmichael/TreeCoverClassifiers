function [ trainingTime, accuracy ] = kFoldCrossValidate_neuralNetwork ...
    ( k, data, labels )
% kFoldCrossValidate_neuralNetwork 
%
% Takes in data, labels and trains a neural network through k-fold cross 
% validation on the dataset. In a neural network, columns are samples and 
% rows are attributes.
%
%   Input:
%       k: the desired number of folds
%       data: the original data
%       labels: the original class labels
%   Output:
%       trainingTime: how long it took to train the model
%       accuracy: the accuracy of the trained model
%
% Note: other measures of accuracy such as a confusion matrix are produced
% by the Matlab Neural Network toolbox and thus do not need to be
% calculated like they do for standard classifiers.

bestTrainingTime = 1000;    % Arbitrarily large number

% Transpose the inputs for the neural network toolbox and convert the
% targets (1-7) into binary vectors
inputs = data(:, :)';
targets = full(ind2vec(labels(:, :)'));

% Calculate chunk size for k-fold validation and divide inputs into k
% chunks
numSamples = size(inputs, 2);
chunkSize = floor(numSamples / k);

% Define vector of hidden layers (length == number of layers, each index is
% number of neurons on that layer)
layers = [50 50 50 50 50];

% Create the network and use specified parameters. Training function can be
% 'trainscg' (default), 'traingd', 'traingda,' or 'traingdx.' Performance
% function can be 'mse,' 'sse,' or 'crossentropy'
net = patternnet(layers, 'traingda', 'sse');
net.divideParam.trainRatio = 90/100;
net.divideParam.valRatio = 10/100;

% Set transfer function on all layers (as it is assigned per-layer)
for currentLayer = 1:size(layers)
    net.layers{currentLayer}.transferFcn = 'logsig';    % logsig, tansig, purelin
end
net.trainParam.epochs = 10;
net.trainParam.max_fail = 10;

% Only set learning rate if using gradient descent function
% (traingd, traingda, traingdx) but default trainscg does not
% utilize it
net.trainParam.lr = 0.1;
    
% 10-fold cross validation for training the network
kFoldRunningAccuracy = 0;
for j = 1:k

    % Divide the inputs into 90% training and 10% testing (the 10% is the
    % k-th chunk of this loop)
    offset = (j - 1) * chunkSize + 1;   % Matlab is 1-indexed
    trainingChunk1 = inputs(:, 1 : offset);
    trainingChunk2 = inputs(:, offset + chunkSize + 1 : end);
    currentTraining = horzcat(trainingChunk1, trainingChunk2);

    trainingTargetChunk1 = targets(:, 1 : offset);
    trainingTargetChunk2 = targets(:, offset + chunkSize + 1 : end);
    currentTargets = horzcat(trainingTargetChunk1, trainingTargetChunk2);

    testChunk = inputs(:, offset : offset + chunkSize);
    testTargetChunk = targets(:, offset : offset + chunkSize);

    % Train the network on this fold of the inputs and see how well it
    % did on the training data
    timeStart = tic;
    [net, tr] = train(net, currentTraining, currentTargets);
    timeElapsed = toc(timeStart);

    if(timeElapsed < bestTrainingTime)
        bestTrainingTime = timeElapsed;
    end

    outputs = net(currentTraining);
    errors = gsubtract(currentTargets, outputs);
    performance = perform(net, currentTargets, outputs);

    % Test the network on the 10% chunk set aside for this fold
    testOutputs = net(testChunk);
%     testIndices = vec2ind(testOutputs);
    [currentAccuracy, confusionMat] = confusion(testTargetChunk, testOutputs);

    kFoldRunningAccuracy = kFoldRunningAccuracy + currentAccuracy;
end

trainingTime = bestTrainingTime;
accuracy = kFoldRunningAccuracy / k;

end

