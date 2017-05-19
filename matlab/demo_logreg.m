clear

% NUTS: example http://arxiv.org/abs/1111.4246

% test on 13 binary classification datasets
load benchmarks.mat;


% parameters for NUTS
n_warm_up = 1000;
train_ratio = 0.8;

N = 100; % number of samples
M = 1; % number of particles
a0 = 1; b0 = 1; % hyper-parameters

for dataset = benchmarks
    bm = eval(char(dataset));
    X = bm.x; Y = bm.t;
    
    X = [ones(size(X,1),1), X]; %bias term
    [n, d] = size(X);
    
    %% random partition
    train_idx = randperm(n, round(train_ratio*n)); test_idx = setdiff(1:n, train_idx);
    Xtrain = X(train_idx, :); Ytrain = Y(train_idx);
    Xtest = X(test_idx, :); Ytest = Y(test_idx);
    train_num = length(train_idx);
    
    % training/testing
    D = d+1; % number of parameters (w & alpha)
    ntrain = size(Xtrain, 1); ntest = size(Xtest, 1);
    log_bayeslr = @(theta) logreg(theta, Xtrain, Ytrain, a0, b0); % log posterior
    log_bayeslr2 = @(theta) dlog_p_lr(theta,Xtrain,Ytrain,train_num,a0,b0);
    
    alpha0 = gamrnd(a0, b0, M, 1);
    theta0 = zeros(M, D);
    for i = 1:M
        theta0(i,:) = [normrnd(0, sqrt((1/alpha0(i))), 1, d), log(alpha0(i))]; % w and log(alpha)
    end
    
    %b1 = log_bayeslr2(theta0);    
    %[a1,a2] = log_bayeslr(theta0);
    
    % NUTS
    tic
    [theta_nuts, ~] = nuts_da(log_bayeslr, N, n_warm_up, randn(1, D));
    toc;

    % evaluation
    [acc, llh] = bayeslr_evaluation(theta_nuts, Xtest, Ytest);  % only need w for evaluation
    
    disp([dataset, size(X,1), size(X,2), acc, llh]);
    
    return;  % EARLY STOP HERE
end