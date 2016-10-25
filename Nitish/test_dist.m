clc;
clear;
close all;
load fmri_words.mat ;

dist_mat = zeros(60,60);
for i = 1:60
  vs = word_features_centered(i,:);
  for j = 1:60
    dist_mat(i,j) = norm(vs-word_features_centered(j,:));
  end
end

[sorted_Y ind] = sort(Y_train);
sorted_X = X_train(ind,:);

intraclass_dist = zeros(60,5);
mean_X = zeros(60,size(sorted_X,2));

for i = 1:60
  current = (i-1)*5+1;
  temp = sorted_X(current,:);
  for j = 1:5
    intraclass_dist(i,j) = norm(temp-sorted_X(current-1+j,:));
  end
  mean_X(i,:) = mean(sorted_X(current:current+4,:),1);
end

interclass_dist = zeros(60,60);
for i = 1:60
  vs = sorted_X((i-1)*5+1,:);
  for j = 1:60
    interclass_dist(i,j) = norm(vs-sorted_X((j-1)*5+1,:));
  end
end