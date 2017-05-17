clear

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample code to test code on bayesian logistic regression
% Implemented from Liu, Q. and Wang, D. (2016) Stein Variational Gradient Descent
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

M = 100; % number of particles

% we partition the data into 80% for training and 20% for testing
train_ratio = 0.8;
max_iter = 6000;  % maximum iteration times

% build up training and testing dataset
load ../data/covertype.mat;
X = covtype(:,2:end); y = covtype(:,1); y(y==2) = -1;

X = [X, ones(size(X,1),1)];  % the bias parameter is absorbed by including 1 as an entry in x
[N, d] = size(X); D = d+1; % w and alpha (prameters)

% building training and testing dataset
train_idx = randperm(N, round(train_ratio*N));  test_idx = setdiff(1:N, train_idx);
X_train = X(train_idx, :); y_train = y(train_idx);
X_test = X(test_idx, :); y_test = y(test_idx);

n_train = length(train_idx); n_test = length(test_idx);

% example of bayesian logistic regression
batchsize = 100; % subsampled mini-batch size
a0 = 1; b0 = .01; % hyper-parameters

% initlization for particles using the prior distribution
alpha0 = gamrnd(a0, b0, M, 1); theta0 = zeros(M, D);
for i = 1:M
    theta0(i,:) = [normrnd(0, sqrt((1/alpha0(i))), 1, d), log(alpha0(i))]; % w and log(alpha)
end



%% Define models
m = M/4;
adverIter = 1;

baseModel = getModel('none',-1);                                   % Base Model
subsetModel = getModel('subset',-1, m);                            % Random Subset
subsetCFModel = getModel('subsetCF',-1, m);                        % Random Subset (CF)
inducedModel = getModel('inducedPoints',-1, m,0);                  % Induced Points
inducedAdver_yModel = getModel('inducedPoints',-1, m,1,adverIter); % Induced Points - update y
inducedAdver_hModel = getModel('inducedPoints',-1, m,2,adverIter); % Induced Points - update h
inducedAdverModel = getModel('inducedPoints',-1, m,3,adverIter);   % Induced Points - update y,h

%% Perform SVGD

% our stein variational gradient descent algorithm %

model = inducedAdverModel;

% Searching best master_stepsize using a development set

tic
dlog_p  = @(theta)dlog_p_lr(theta, X_train, y_train); % returns the first order derivative of the posterior distribution
for i = 1:5
theta_svgd = svgd(theta0, dlog_p, max_iter, model);
end
time = toc;

% evaluation
[acc_svgd, llh_svgd] = bayeslr_evaluation(theta_svgd, X_test, y_test);
fprintf('Result of SVGD: testing accuracy: %f; testing loglikelihood: %f, running time: %fs\n', acc_svgd, llh_svgd, time);
