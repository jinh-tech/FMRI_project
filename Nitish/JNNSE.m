clc;
clear;
close all;

K = 800;    %CHANGE THIS VALUE 

global no_train_samples no_actual_features no_words no_word_features;
global X_train Y_feature;

load fmri_words.mat;

[no_train_samples, no_actual_features] = size(X_train);     % [300 21764]   
[no_words, no_word_features] = size(word_features_centered);    %[60 218]

Y_feature = zeros(no_train_samples,no_word_features);   %[300 218]
for i = 1: no_train_samples
    Y_feature(i,:) = word_features_centered(Y_train(i),:); 
end
% K = no of latent vector (l)

A  = randn(no_train_samples,K);
Da = randn(no_actual_features,K);
Db = randn(no_word_features, K);

lambda = 10; 
num_iters = 10;

for iter=1:num_iters
    % update Da, one row at a time
    I = lambda*eye(K);
    temp = A'*A;
    for n=1:no_actual_features
        Da(n,:) = ((temp + I)\A'*X_train(:,n))';
    end
    % update Db, one row at a time
    for n=1:no_word_features
        Db(n,:) = ((temp + I)\A'*Y_feature(:,n))';
    end
    % update A, one row at a time
    X_new = [X_train, Y_feature];
    D_new = [Da; Db];
    temp = D_new'*D_new;
    for r=1:no_train_samples
        A(r,:) = (temp + I)\D_new'*X_new(r,:)';
    end
    fprintf('loop %d\n', iter);
end

W = X_train\A;  % 21K X 400
result=X_test*W;   % 60 X 400
count =0;
for i = 1:60
    sum = zeros(1,2);
    for j = 1:2
        for k = 1:300
            if (Y_test(i,j)==Y_train(k))
                sum(1,j)=sum(1,j)+norm(result(i,:)-A(Y_train(k)),2);
            end
        end
        sum(1,j) = sum(1,j)/5;
    end
    [a,b] = max(sum(1,:));
    if(b==1)
        count = count+1;
    end
end
count
