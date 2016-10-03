tic;
clear;
clc;
close all;

global no_train_samples no_actual_features no_words no_word_features no_reduced_features;
global X_train Y_feature;

load fmri_words.mat;

[no_train_samples, no_actual_features] = size(X_train);     % [300 21764]   
[no_words, no_word_features] = size(word_features_centered);    %[60 218]

Y_feature = zeros(no_train_samples,no_word_features);   %[300 218]
for i = 1: no_train_samples
    Y_feature(i,:) = word_features_std(Y_train(i),:); 
end

pred_accuracy = zeros(1,20);
recon_accuracy = zeros(1,20);

for no_reduced_features = 1000:1000:20000

    reduced_training_set = rand(no_train_samples,no_reduced_features).*2;      %[300 k]
    basis_matrix = rand(no_reduced_features,no_actual_features).*2;    %[k 21764]
    weights = rand(no_reduced_features,no_word_features);    %[k 218]

    % vectorized_matrix = [reduced_training_set,basis_matrix,weights]
    vectorized_matrix = reshape(reduced_training_set,[1 no_train_samples*no_reduced_features]);
    vectorized_matrix = [vectorized_matrix,reshape(basis_matrix,[1 no_reduced_features*no_actual_features]) ];
    vectorized_matrix = [vectorized_matrix,reshape(weights,[1,no_reduced_features*no_word_features])];

    %---------Gradient-descent ---------------------
    %options = optimoptions('fminunc','Algorithm','trust-region','SpecifyObjectiveGradient',true);
    solution = fminsearch(@svdm_loss_func,vectorized_matrix);

    %-----------De-vectorizing solution-------------
    Z = reshape(vectorized_matrix(1:no_train_samples*no_reduced_features),[no_train_samples no_reduced_features]);
    vectorized_matrix = vectorized_matrix(no_train_samples*no_reduced_features+1:end);

    W = reshape(vectorized_matrix(1:no_reduced_features*no_actual_features),[no_reduced_features no_actual_features]);
    vectorized_matrix = vectorized_matrix(no_reduced_features*no_actual_features+1:end);

    theta = reshape(vectorized_matrix(1:no_reduced_features*no_word_features),[no_reduced_features,no_word_features]);

    %---------Calculating accuracies---------------
    pred_accuracy(i) = norm(Y_feature - Z*theta)
    recon_accuracy(i) = norm(X_train - Z*W)
    
end

plot(1:20,pred_accuracy);
plot(1:20,recon_accuracy);

toc;