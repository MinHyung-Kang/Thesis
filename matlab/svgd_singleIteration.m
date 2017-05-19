function  [theta, gradInfo, opts] = svgd_singleIteration(theta, dlog_p_ori, gradInfo, opts)
%%%%%%%%
% Bayesian Inference via Stein Variational Gradient Descent (Single Iteration)
% Implemented from Liu, Q. and Wang, D. (2016) Stein Variational Gradient Descent

% Input:
%   -- theta0: initialization of particles, m * d matrix (m is the number of particles, d is the dimension)
%   -- dlog_p: function handle of first order derivative of log p(x)
%   -- max_iter: maximum iterations
%   -- opts : other options for training

% output:
%   -- theta: a set of particles after update
%   -- gradInfo: include historical gradient information
%   -- opts : other options for training
%%%%%%%%

fudge_factor = 1e-6;
historical_grad = gradInfo.historical_grad;
y_historical_grad = gradInfo.y_historical_grad;
h_historical_grad = gradInfo.h_historical_grad;

switch lower(opts.method)
case 'adagrad'
    % Set the batch as same as the one that was used to update y and h
    dlog_p = @(val)dlog_p_ori(val, opts.currentSeed);
    [grad, ksdInfo] = KSD_KL_gradxy(theta, dlog_p, opts.kernel_opts);   %\Phi(theta)
    gradInfo.ksdInfo = ksdInfo;
    [adam_grad,historical_grad] = getAdamUpdate(grad, historical_grad, opts.master_stepsize, opts.auto_corr, fudge_factor);
    theta = theta + adam_grad; % update

    gradInfo.historical_grad = historical_grad;
    opts.currentSeed = opts.nextSeed;
    dlog_p = @(val)dlog_p_ori(val, opts.currentSeed);
    
    if strcmp(opts.kernel_opts.method,'inducedPoints') == 1 && opts.kernel_opts.adver > 0
        Y = theta(randi([1,size(theta,1)],opts.kernel_opts.m,1),:);
        h = -1;
        opts.kernel_opts.Y = Y;
        opts.kernel_opts.h = h;
        Sqx = dlog_p(theta);

        for adverInd = 1:opts.kernel_opts.adverIter
            % If using adversairal updates for y
            if opts.kernel_opts.adver == 1 || opts.kernel_opts.adver == 3
                %[yGrad, ~] = inducedKernel_grady(theta, Y, ksdInfo.Sqx, opts.kernel_opts);
                [yGrad, ~] = inducedKernel_grady(theta, Y, Sqx, opts.kernel_opts);
                [adam_yGrad,y_historical_grad] = getAdamUpdate(yGrad, y_historical_grad, opts.master_stepsize, opts.auto_corr, fudge_factor);
                Y = Y + adam_yGrad; % update
                opts.kernel_opts.Y = Y;
            end

            % If using adversairal updates for h
            if opts.kernel_opts.adver == 2 || opts.kernel_opts.adver == 3
                %[hGrad, info] = inducedKernel_gradh(theta, Y, ksdInfo.Sqx, opts.kernel_opts);
                [hGrad, info] = inducedKernel_gradh(theta, Y, Sqx, opts.kernel_opts);
                if h == -1; h = info.bandwidth; end
                [adam_hGrad,h_historical_grad] = getAdamUpdate(hGrad, h_historical_grad, opts.master_stepsize, opts.auto_corr, fudge_factor);
                h = h + adam_hGrad; % update
                opts.kernel_opts.h = h;

            end

        end

        gradInfo.y_historical_grad = y_historical_grad;
        gradInfo.h_historical_grad = h_historical_grad;
    end


otherwise
    error('wrong method');
end
end
