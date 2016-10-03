function [loss ] = svdm_loss_func(vectorized_matrix)
 
global no_train_samples no_actual_features no_word_features no_reduced_features;
global X_train Y_feature;

%----------De-vectorizing------------------------
% vectorized_matrix = [Z,W,theta]

Z = reshape(vectorized_matrix(1:no_train_samples*no_reduced_features),[no_train_samples no_reduced_features]);
vectorized_matrix = vectorized_matrix(no_train_samples*no_reduced_features+1:end);

W = reshape(vectorized_matrix(1:no_reduced_features*no_actual_features),[no_reduced_features no_actual_features]);
vectorized_matrix = vectorized_matrix(no_reduced_features*no_actual_features+1:end);

theta = reshape(vectorized_matrix(1:no_reduced_features*no_word_features),[no_reduced_features,no_word_features]);

loss = norm(X_train-Z*W)^2 + norm(Y_feature - Z*theta)^2;

% theta_grad = zeros(size(theta));
% Z_grad = zeros(size(Z));
% W_grad = zeros(size(W));
% 
% %------------------W _grad------------------------
% 
% for i = 1:no_actual_features
%     temp = X_train(:,i) - Z*W(:,i);
%     tempsum = zeros(no_reduced_features,1);
%     for j = 1:no_train_samples
%         tempsum = tempsum + Z(j,:)'.*temp(j);
%     end
%     W_grad(:,i) = -tempsum;
% end    
% 
% %----------------Theta_grad--------------------------
% 
% for i = 1:no_word_features
%     temp = Y_feature(:,i) - Z*theta(:,i);
%     tempsum = zeros(no_reduced_features,1);
%     for j = 1:no_train_samples
%         tempsum = tempsum + Z(j,:)'.*temp(j);
%     end
%     theta_grad(:,i) = - tempsum;    
% end
% 
% %------------------Z-grad_---------------------------
% 
% for i = 1:no_train_samples
%     temp = X_train(i,:) - Z(i,:)*W;
%     temp1 = Y_feature(i,:) - Z(i,:)*theta;
%     tempsum = zeros(1,no_reduced_features);
%     for j = 1:no_actual_features
%         tempsum = tempsum + W(:,j)'.*temp(j);
%     end
%     tempsum1 = zeros(1,no_reduced_features);
%     for j = 1:no_word_features
%         tempsum1 = tempsum1 + theta(:,j)'.*temp1(j);
%     end
%     Z_grad(i,:) = -tempsum - tempsum1;
%     
% end
% 
% %-----------Vectorizing gradients------------------
% 
% gradient = reshape(Z_grad,[1,no_train_samples*no_reduced_features]);
% gradient = [gradient,reshape(W_grad,[1,no_reduced_features*no_actual_features])];
% gradient = [gradient,reshape(theta_grad,[1,no_reduced_features*no_word_features])];

