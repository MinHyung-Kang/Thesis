%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sample code to test mean/variance estimation of SVGD
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all;
rng(5);
% Define a random Gaussian distribution
d = 2;
mu = d*randn(1,d);
Crand = rand(d,d);
sigma = 2*(Crand + Crand' + d * eye(d));

% To draw contour of the distribution
x1 = -8:0.1:8; x2 = -8:0.1:8;
[Grid.X1,Grid.X2] = meshgrid(x1,x2);
[grid1, grid2] = size(Grid.X1);
X1_2d = reshape(Grid.X1, grid1 * grid2,1);
X2_2d = reshape(Grid.X2, grid1 * grid2,1);
Grid.pVal = reshape(mvnpdf([X1_2d X2_2d], mu, sigma), grid1, grid2);

evalTheta = @(points)evalPoints(points, mu, sigma);
dlog_p = @(X)((X - repmat(mu,size(X,1),1)) * -inv(sigma));

%% Options

fudge_factor = 1e-6;
% 
M = 50;
maxIter = 800;  % maximum iteration times
numTimeSteps = 10; % How many timesteps we want to shows
maxTrial = 20;% How many trials we want to average over
timeStepUnit = maxIter / numTimeSteps;
m = 20;
adverIter = 10;


% Common options
baseModel = getModel('none',-1);                                   % Base Model
subsetModel = getModel('subset',-1, m);                            % Random Subset
subsetCFModel = getModel('subsetCF',-1, m);                        % Random Subset (CF)
inducedModel = getModel('inducedPoints',-1, m,0,-1,0,false);       % Induced Points
inducedAdver_yModel = getModel('inducedPoints',-1, m,1,adverIter); % Induced Points - update y
inducedAdver_hModel = getModel('inducedPoints',-1, m,2,adverIter); % Induced Points - update h
inducedAdverModel = getModel('inducedPoints',-1, m,3,adverIter);   % Induced Points - update y,h

% Induced Points - update y,h; no Additional methods used
inducedAdverModel_basic = getModel('inducedPoints',-1, m,3,adverIter,0,false);   
% Induced Points - update y,h; use uStats
inducedAdverModel_justU = getModel('inducedPoints',-1, m,3,adverIter,0,true);  
% Induced Points - update y,h; use regularization
inducedAdverModel_justReg = getModel('inducedPoints',-1, m,3,adverIter,0.1,false);   

% 1. Compare basic methods
modelOpts = {baseModel, subsetModel, subsetCFModel, inducedModel, inducedAdverModel};
algNames= {'base','subset','subsetCF','IP', 'AIP(10)'};

% 2. Compare y vs h updates
% inducedAdver_yModel = getModel('inducedPoints',-1, m,1,adverIter,0,false); % Induced Points - update y
% inducedAdver_hModel = getModel('inducedPoints',-1, m,2,adverIter,0,false); % Induced Points - update h
% inducedAdverModel = getModel('inducedPoints',-1, m,3,adverIter,0,false);   % Induced Points - update y,h
% modelOpts = {inducedModel, inducedAdver_yModel, inducedAdver_hModel, inducedAdverModel};
% algNames= {'base','y-update','h-update','y,h-update'};

% 3.  Compare regularization vs uStat
% inducedAdverModel_justReg = getModel('inducedPoints',-1, m,3,adverIter,0.1,false);   
% inducedAdverModel = getModel('inducedPoints',-1, m,3,adverIter,0.1,true);   % Induced Points - update y,h
% modelOpts = {inducedAdverModel_basic, inducedAdverModel_justReg,inducedAdverModel_justU, inducedAdverModel};
% algNames= {'basic','reg=0.1','uStat', 'reg=0.1,uStat'};


% 4. Compare y vs h updates using regularization and ustat
% inducedAdver_yModel = getModel('inducedPoints',-1, m,1,adverIter,0.1,true); % Induced Points - update y
% inducedAdver_hModel = getModel('inducedPoints',-1, m,2,adverIter,0.1,true); % Induced Points - update h
% inducedAdverModel = getModel('inducedPoints',-1, m,3,adverIter,0.1,true);   % Induced Points - update y,h
% modelOpts = {inducedModel, inducedAdver_yModel, inducedAdver_hModel, inducedAdverModel};
% algNames= {'base','y-update','h-update','y,h-update'};


% 4. Compare y vs h updates using just uStat
% inducedAdver_yModel = getModel('inducedPoints',-1, m,1,adverIter,0,true); % Induced Points - update y
% inducedAdver_hModel = getModel('inducedPoints',-1, m,2,adverIter,0,true); % Induced Points - update h
% inducedAdverModel = getModel('inducedPoints',-1, m,3,adverIter,0,true);   % Induced Points - update y,h
% modelOpts = {inducedModel, inducedAdver_yModel, inducedAdver_hModel, inducedAdverModel};
% algNames= {'base','y-update','h-update','y,h-update'};


%% Perform SVGD
numModels = length(modelOpts);   % How many models we want to try

tVal = zeros(numModels, numTimeSteps);
RMSE_mu = zeros(numModels, numTimeSteps);
RMSE_cov = zeros(numModels, numTimeSteps);
valNames = {'t','RMSEmu','RMSEcov'};

% For each trial
mu0 = [6,-4];
for trialInd = 1:maxTrial
    % Common starting points
    theta0 = mvnrnd(mu0,2*eye(2),M);

    % Plot only if we are running the code once
    if maxTrial == 1
        figure('Position', [500, 100, 800, 600]);
        plotW = 2;
        plotH = ceil(numModels/plotW);
    end

    % For each model
    for modelInd = 1:numModels
        theta = theta0;
        modelOpt = modelOpts{modelInd};
        h = modelOpt.kernel_opts.h;
        
        if modelOpt.kernel_opts.adver > 0
            Y = theta(randi([1,M], modelOpt.kernel_opts.m,1),:);
            modelOpt.kernel_opts.Y = Y;
        end

        % Plot the initial contour
        if (maxTrial == 1)
            subplot(plotH, plotW, modelInd);
            hold on;
            contour(Grid.X1, Grid.X2, Grid.pVal, 'g');
            plot_h = plot(theta0(:,1),theta0(:,2),'b*');

            % If performing adversarial updates
            if modelOpt.kernel_opts.adver > 0
                plot_y = plot(Y(:,1),Y(:,2),'ro');
            end
            title(algNames{modelInd});
            drawnow;
        end

        historical_grad = 0;
        y_historical_grad = 0;
        h_historical_grad = 0;
        timePassed = 0;

        % Iteration
        for iter = 1:maxIter
            % Get update
            timeStart = tic;

            % 1. Update theta
            [grad, ksdInfo] = KSD_KL_gradxy(theta, dlog_p, modelOpt.kernel_opts);   %\Phi(theta)
            [adam_grad,historical_grad] = getAdamUpdate(grad, historical_grad, ...
                modelOpt.master_stepsize, modelOpt.auto_corr, fudge_factor);
            theta = theta + adam_grad; % update

            % Show theta update
            if (maxTrial == 1) && mod(iter, timeStepUnit) == 0
                pause(0.1);
                delete(plot_h);
                plot_h = plot(theta(:,1),theta(:,2),'b*');
                drawnow;
            end
            
            % 2. Make updates for y and h
            if modelOpt.kernel_opts.adver > 0
                %y_historical_grad = 0;
                %h_historical_grad = 0;
                Y = theta(randi([1,M], modelOpt.kernel_opts.m,1),:);
                h = -1;
                modelOpt.kernel_opts.h = h;
                modelOpt.kernel_opts.Y = Y;
                
                
                for adverInd = 1:modelOpt.kernel_opts.adverIter
                    % If using adversarial updates for y
                    if modelOpt.kernel_opts.adver == 1 || modelOpt.kernel_opts.adver == 3

                        [yGrad, ~] = inducedKernel_grady(theta, Y, ksdInfo.Sqx, modelOpt.kernel_opts);
                        [adam_yGrad,y_historical_grad] = getAdamUpdate(yGrad, y_historical_grad, ...
                            modelOpt.master_stepsize, modelOpt.auto_corr, fudge_factor);
                        Y = Y + adam_yGrad; % update 
                        modelOpt.kernel_opts.Y = Y;                
                    end

                    % If using adversairal updates for h
                    if modelOpt.kernel_opts.adver == 2 || modelOpt.kernel_opts.adver == 3

                        [hGrad, info] = inducedKernel_gradh(theta, Y, ksdInfo.Sqx, modelOpt.kernel_opts);

                        if h == -1; h = info.bandwidth; end

                        [adam_hGrad,h_historical_grad] = getAdamUpdate(hGrad, h_historical_grad, ...
                            modelOpt.master_stepsize, modelOpt.auto_corr, fudge_factor);
                        h = h + adam_hGrad; % update 
                        modelOpt.kernel_opts.h = h; 
                    end

                    % Show Y updates
                    if (maxTrial == 1) && mod(iter, timeStepUnit) == 0
                        pause(0.1);
                        delete(plot_y);
                        plot_y = plot(Y(:,1),Y(:,2),'ro');
                        drawnow;
                    end
                end

            end

            timePassed = timePassed + toc(timeStart);
            % Print and evaluate at current step
            if mod(iter, timeStepUnit) == 0
                [RMSE_mu_val, RMSE_cov_val] = evalTheta(theta);

                iterUnit = iter / timeStepUnit;
                fprintf('Evaluating (%d/%d trials) : Model (%d/%d) - Iteration (%d/%d)\n', ...
                        trialInd, maxTrial, modelInd, numModels, iter, maxIter);
                tVal(modelInd,iterUnit) = tVal(modelInd,iterUnit) + timePassed / maxTrial;
                RMSE_mu(modelInd,iterUnit) = RMSE_mu(modelInd,iterUnit) + RMSE_mu_val / maxTrial;
                RMSE_cov(modelInd,iterUnit) = RMSE_cov(modelInd,iterUnit) + RMSE_cov_val / maxTrial;
            end

        end

    end

end


results = struct(valNames{1},tVal,valNames{2},RMSE_mu,valNames{3}, RMSE_cov);
%% Plot results
%% Plot results
figure;
colOpts = {'h-','o-','*-','.-','x-','s-','d-','^-','v-','p-','h-','>-','<-'};
titleNames = {'Total Time', 'log(RMSE(mu))', 'log(RMSE(cov))'};
plotTime = timeStepUnit * (1:numTimeSteps);

% Time 
subplot(2,3,1);
hold on;
result = results.(valNames{1});

for i = 1:numModels
   plot(plotTime, result(i,:),colOpts{i});
end
title(sprintf('%s',valNames{1}));
xlabel('Iterations');
ylabel(valNames{1});

% Time (except AIP)
subplot(2,3,2);
hold on;
result = results.(valNames{1});

for i = 1:numModels-1
   plot(plotTime, result(i,:),colOpts{i});
end
title(sprintf('%s (except AIP)',valNames{1}));
xlabel('Iterations');
ylabel(valNames{1});

% Accuracy
subplot(2,3,4);
hold on;
result = results.(valNames{2});

for i = 1:numModels
   plot(plotTime, result(i,:),colOpts{i});
end
title(sprintf('%s',valNames{2}));
xlabel('Iterations');
ylabel(valNames{2});

% LLh
subplot(2,3,5);
hold on;
result = results.(valNames{3});
handles = zeros(1, numModels);
for i = 1:numModels
   handles(i) = plot(plotTime, result(i,:),colOpts{i});
end
title(sprintf('%s',valNames{3}));
xlabel('Iterations');
ylabel(valNames{3});


subplot(2,3,6);
axis off;
leg1 = legend(handles,algNames, 'Orientation','vertical');
%set(leg1, 'Position',[0.7 0.3 0 0]);
set(leg1, 'Position',[0.8 0.5 0.05 0.05]);


% %% Plot results
% figure;
% colOpts = {'h-','o-','*-','.-','x-','s-','d-','^-','v-','p-','h-','>-','<-'};
% titleNames = {'Total Time', 'log(RMSE(mu))', 'log(RMSE(cov))'};
% plotTime = timeStepUnit * (1:numTimeSteps);
% 
% for j = 1:3
%     subplot(2,3,j);
%     hold on;
%     handles = zeros(1, numModels-1);
%     result = results.(valNames{j});
% 
%     for i = 1:numModels
%        handles(i) = plot(plotTime, result(i,:),colOpts{i});
%     end
%     %title(sprintf('%s (Avg of %d trials)',valNames{j}, maxTrial));
%     title(sprintf('%s',valNames{j}));
%     xlabel('Iterations');
%     ylabel(valNames{j});
% end
% 
% subplot(1,4,4);
% axis off;
% leg1 = legend(handles,algNames, 'Orientation','vertical');
% %set(leg1, 'Position',[0.7 0.3 0 0]);
% set(leg1, 'Position',[0.8 0.5 0.05 0.05]);

%% Auxiliary functions

function [RMSE_mu, RMSE_cov] = evalPoints(points, mu, cov)
    [mu_hat, mu_cov] = estimateGaussianMean(points);

    RMSE_mu = log10(RMSE(mu_hat, mu));
    RMSE_cov = log10(RMSE(mu_cov, cov));
end

function val = RMSE(yReal,yhat)
    val = sqrt(mean((yReal(:) - yhat(:)).^2));
end

function [mu_hat, mu_cov] = estimateGaussianMean(points)
    mu_hat = mean(points,1);
    points_n = size(points,1);
    covMat = (points - repmat(mu_hat,points_n,1));
    mu_cov = covMat' * covMat / points_n;
end
