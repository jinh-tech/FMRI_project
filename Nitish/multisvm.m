clc;
clear;
load fmri_words.mat

num_class = 60;
svm_model = cell(60,1);
scores = zeros(size(X_test,1),60);

for i = 1:num_class
    index = (Y_train==i);
    svm_model{i} = fitcsvm(X_train,index);
end

for i = 1:num_class
    [~,temp] = predict(svm_model{i},X_test);
    scores(:,i) = temp(:,2);
end

count = 0;
for i = 1:size(X_test,1)
    if scores(i,Y_test(i,1))>scores(i,Y_test(i,2))
        count = count + 1;
    end
end

count