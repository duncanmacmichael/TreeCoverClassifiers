%--------------------------------------------------------------------------
% Tree Cover Classifiers
% Author: Duncan MacMichael
% 
% The goal of this Tree Cover Classifiers project is to apply different
% classifier models to Forest Cover Type dataset, acquired from UC Irvine
% Machine Learning Repository. It is a multivariate dataset with 55
% columns. Each of the 581,012 rows is a sample that represents a 30x30
% meter area, with attributes gathered from cartographic observations in
% four different wilderness areas. The data is clean - that is, there are
% no missing values or malformed data. The class label is the final column,
% the Cover Type (or dominant tree type in that sample). The columns are:
%   Elevation in meters
%   Aspect in degrees azimuth
%   Slope in degrees
%   Horizontal distance to hydrology (surface water features)
%   Vertical distance to hydrology
%   Horizontal distance to roadways
%   Hillshade at 9 AM (0 to 255 index), summer solstice
%   Hillshade at 12 PM (0 to 255 index), summer solstice
%   Hillshade at 3 PM (0 to 255 index), summer solstice
%   Horizontal distance to fire points (nearest wildfire ignition points)
%   Wilderness area designation (4 binary columns, one for each area)
%   Soil Type (40 binary columns, 0 = absence or 1 = presence)
%   Cover Type (7 types, integers from 1 to 7)
%--------------------------------------------------------------------------

% Preprocess the data. Collapses binary columns, normalizes data. Returns
% as many columns as specified by the second 'numPrincipalComponents'
% argument to preProcessData.
[data, labels] = preProcessData('covtype_wildernessArea1.csv', 12);

%--------------------------------------------------------------------------
% Train standard classifiers and get their accuracy/results.
%--------------------------------------------------------------------------

% Train a Naive Bayes ('naiveBayes'), Decision Tree ('decisionTree'), or
% K-Nearest Neighbor ('knn') classifier and get the accuracy/results
% [trainingTime, predictionTime, accuracy, confusionMatrix, order ] = ...
%     kFoldCrossValidate_standardClassifiers(10, data, labels, 'knn');
% showResults(trainingTime, predictionTime, accuracy, confusionMatrix, order);

%--------------------------------------------------------------------------
% Train a neural network and get its accuracy/results.
%--------------------------------------------------------------------------

inputs = data(:, :)';
targets = full(ind2vec(labels(:, :)'));
% [trainingTime, accuracy] = kFoldCrossValidate_neuralNetwork(10, data, ...
%     labels);
% fprintf('Neural Network Accuracy was: %f%% and training time was: %5.2f seconds.\n', ...
%     100 * accuracy, trainingTime);