function[count] = linear_regression_nnmf(X,Word_train,Test_data,word_features_centered,Y_test)

% B=zeros(21764,218);
% for i=1:218
%     B(:,i)=X_train\Word_train(:,i);
% end
X = [ones(size(X,1),1) X];
Test_data = [ones(size(Test_data,1),1) Test_data];
B = X\Word_train;
%-----------------------------------------------------
wtest=Test_data*B;
count=0;
for i=1:60
    if(norm(wtest(i,:)-word_features_centered(Y_test(i,1),:))<norm(wtest(i,:)-word_features_centered(Y_test(i,2),:)))
         count=count+1;
    end
end

