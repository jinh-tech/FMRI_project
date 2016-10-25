clc;
clear;
load fmri_words.mat;

acc = zeros(20,1);
step = 500;
Word_train = zeros(300,218);
for i=1:300
      Word_train(i,:) = word_features_centered(Y_train(i),:);
end

offset = min(min(X_train));
%Word_train = repmat(Word_train,[20 1]);
X = X_train + offset;
X = repmat(X,[20 1]);

Y_test = repmat(Y_test,[100 1]);
Test = X_test + offset;
Test = repmat(Test,[100 1]);

for k = 500:step:5500
        
    [W,H] = nnmf(X,k);
    [Test_data,~] = nnmf(Test,k,'h0',H);
    W = W(1:300,:);
    Test_data = Test_data(1:60,:);
    acc(k/step) = linear_regression_nnmf(W,Word_train,Test_data,word_features_centered,Y_test);
end
