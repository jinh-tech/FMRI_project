load('fmri_words.mat');
Word_train=zeros(300,218);
for i=1:300
      Word_train(i,:)=word_features_centered(Y_train(i),:);
end

% B=zeros(21764,218);
% for i=1:218
%     B(:,i)=X_train\Word_train(:,i);
% end

B = X_train\Word_train;

wtest=X_test*B;
ans=zeros(60,1);
count=0;
for i=1:60
    if(norm(wtest(i,:)-word_features_centered(Y_test(i,1),:))<norm(wtest(i,:)-word_features_centered(Y_test(i,2),:)))
         count=count+1;
    end
end
