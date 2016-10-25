% Implementing loss function (  |X-ZW|^2 + lambda|Z|^2  ) 
% X - training data , Z = reduced training data , W  = basis 

clc;
clear;
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