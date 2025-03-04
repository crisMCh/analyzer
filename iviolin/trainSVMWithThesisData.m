clear all
load('OctaveFeatSet.mat')
trainData = featSet;
% load('trainLabels.mat')
trainLabels(1:15,1) = 1; % Good cases 
trainLabels(17:30,1) = 0; % Bad cases

model = svmtrain(trainLabels, trainData, '-t 2');

[predicted_label, accuracy, decision_valuesprob_estimates] = ...
svmpredict(trainLabels(:,1), ...
    trainData(:,:), ...
    model)
    
    
save('svmModelOctave.mat', 'model')

clear all

load('svmModelOctave.mat');