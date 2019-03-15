function [  ] = showResults( trainingTime, predictionTime, accuracy, ...
    confusionMat, order )
% showResults
%
% Displays the training results of a trained classifier model.
%   Input:
%       trainingTime: how long it took to train the model
%       predictionTime: how long it took the trained model to predict class
%           labels on new data
%       accuracy: the accuracy of the trained model
%       confusionMatrix: a confusion matrix for the trained model
%       order: a vector of the order of the observed class labels (for use
%           with the confusion matrix to label what the axes are)       
%   Output:
%       none

fprintf('Training time was: %5.2f seconds.\n', trainingTime);
fprintf('Prediction time was: %5.2f seconds.\n', predictionTime);
fprintf('Accuracy was: %5.2f%%.\n', accuracy);

% Calculate precision, recall, and f-scores based on the confusion matrix.
% The following lines of code for precision, recall, and f-score calculation 
% are borrowed from Stack Overflow:
% http://stackoverflow.com/questions/22915003/is-there-any-function-to-calculate-precision-and-recall-using-matlab

% Precision = exactness. (true positives) / (true positives + false positives)
% Recall = completeness. (true positives) / (true positives + false negatives)
% F-score = harmonic mean of precision and recall. (2 * precision * recall) / (precision + recall)

function modelPrecision = precision(confusionMat)
    modelPrecision = diag(confusionMat) ./ sum(confusionMat, 2);
end

function modelRecall = recall(confusionMat)
        modelRecall = diag(confusionMat) ./ sum(confusionMat, 1)';
end

function modelFScores = fScores(confusionMat)
        modelFScores = 2 * (precision(confusionMat) .* ...
            recall(confusionMat)) ./ (precision(confusionMat) + ...
            recall(confusionMat));
end

precisionToPrint = mean(precision(confusionMat));
recallToPrint = mean(recall(confusionMat));
fScoresToPrint = mean(fScores(confusionMat));

fprintf('Precision was: %5.5f\n', precisionToPrint);
fprintf('Recall was: %5.5f\n', recallToPrint);
fprintf('F-score was: %5.5f\n', fScoresToPrint);

disp(' ');
disp(order');
disp(confusionMat);

end

