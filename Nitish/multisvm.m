clc;
clear;
load fmri_words.mat
%load A.mat
%load W.mat
load final1.mat

accuracy = [];

for iter = 25:25:4000
    
X_test_ind = X_test(:,final(1:iter));
X_train_ind = X_train(:,final(1:iter));
%X_test = X_test*W;
%X_train = A;


num_class = 60;
svm_model = cell(60,1);
scores = zeros(size(X_test,1),60);

for i = 1:num_class
    index = (Y_train==i);
    svm_model{i} = fitcsvm(X_train_ind,index);   %fitcsvm(X_train,index);
end

for i = 1:num_class
    [~,temp] = predict(svm_model{i},X_test_ind);
    scores(:,i) = temp(:,2);
end

count = 0;
for i = 1:size(X_test_ind,1)
    if scores(i,Y_test(i,1))>scores(i,Y_test(i,2))
        count = count + 1;
    end
end

accuracy = [accuracy count];
end

plot(25:25:4000,accuracy);