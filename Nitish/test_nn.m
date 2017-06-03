clear;
clc;
close all;
load fmri_words.mat;

[N,M] = size(X_train);
error = zeros(1,31);
j=1;

for i = 400:10:400
i
U = rand(N,i);
V = rand(M,i);
    for iter = 1:10
        new_V = (V'*V)\V';
        U = (new_V*X_train')';
        
        new_U = (U'*U)\U';
        V = (new_U*X_train)';
    end
error(j) = norm(X_train-U*V','fro')/sqrt(N*M);
j = j+1;    
end


plot(1:30,error);