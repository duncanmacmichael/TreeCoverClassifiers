% Train a neural network based on the tree cover type dataset
% Author: Duncan MacMichael

% Set up parallel pool of CPU workers (requires Parallel Computing Toolbox)
% Note: GPU computation is possible but not feasible, as the default output
% processing function, 'removeconstantrows', is not compatible with GPUs.
% Changing that to 'mapminmax' does allow for GPU parallelism, but results
% in instant failure with the network hitting maximum gradient after one
% iteration. Therefore, for parallel computation, this network only uses
% multiple CPU cores.
pool = parpool;

% Read in the data, preprocess it, and transpose it for the neural network
[data, labels] = preProcessData('covtype.csv', 12);
inputs = data(:, :)';
targets = full(ind2vec(labels(:, :)'));

% After applying PCA, determine which attributes have most impact
% Parameters: inputs, labels, K-nearest neighbors, number of observations
% [ranked, weights] = relieff(data, labels, 100, 'updates', 10000);
% disp('ranked = ');
% disp(ranked);
% disp('weights = ');
% disp(weights);

% Set the training function and performance function
trainFcn = 'trainrp';   % resilient backpropagation
perfFcn = 'crossentropy';        

% Define the layer vector and create the Pattern Recognition Network
layers = [100 100 100 100 100];
net = patternnet(layers, trainFcn, perfFcn);

% Set transfer function on all layers (as it is assigned per-layer)
for currentLayer = 1:size(layers)
    net.layers{currentLayer}.transferFcn = 'logsig';    % logsig, tansig, purelin
end
net.trainParam.epochs = 1000;
net.trainParam.max_fail = 100;

% Only set learning rate if using gradient descent function
% (traingd, traingda, traingdx) but default trainscg does not
% utilize it
net.trainParam.lr = 0.1;

% Set up division of data for Training, Validation, and Testing
net.divideParam.trainRatio = 50/100;
net.divideParam.valRatio = 25/100;
net.divideParam.testRatio = 25/100;

% Train the network
[trainedNet, trainingRecords] = train(net, inputs, targets, ...
    'useParallel', 'yes');

% Test the network
predictions = trainedNet(inputs, 'useParallel', 'yes');
errors = gsubtract(targets, predictions);
performance = perform(trainedNet, targets, predictions);
targetIndicies = vec2ind(targets);
predictionIndices = vec2ind(predictions);
percentErrors = sum(targetIndicies ~= predictionIndices)/numel(targetIndicies);

% Shut down the parallel pool
delete(gcp('nocreate'));

% View the network
view(net);

% Manually show certain plots (nntraintool usually facilitates this)
%figure, plotperform(trainingRecords);
%figure, plottrainstate(trainingRecords);
%figure, ploterrhist(errors);
%figure, plotconfusion(targets, predictions);
%figure, plotroc(targets, predictions);