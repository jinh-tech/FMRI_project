clear;
clc;
close all;
load fmri_words.mat;

cvp = cvpartition(Y_train,'kfold',5);
numvalidsets = cvp.NumTestSets;

n = length(Y_train);
lambdavals = linspace(0,20,20)/n;
lossvals = zeros(length(lambdavals),numvalidsets);

for i = 1:length(lambdavals)
    for k = 1:numvalidsets
       
        X = X_train(cvp.training(k),:);
        y = Y_train(cvp.training(k),:);
        Xvalid = X_train(cvp.test(k),:);
        yvalid = Y_train(cvp.test(k),:);
        nca1 = fscnca(X,y,'FitMethod','exact','Solver','sgd','Lambda',lambdavals(i),'IterationLimit',30,'GradientTolerance',1e-4,'Standardize',true);
        lossvals(i,k) = loss(nca1,Xvalid,yvalid,'LossFunction','classiferror');
    end
end

meanloss = mean(lossvals,2);

[~,idx] = min(meanloss); % Find the index
bestlambda = lambdavals(idx); % Find the best lambda value
bestloss = meanloss(idx);

nca1 = fscnca(Xtrain,ytrain,'FitMethod','exact','Solver','sgd','Lambda',bestlambda,'Standardize',true,'Verbose',1);
[~,index] = sort(nca1.FeatureWeights,'descend');
final = index(1:4000);
