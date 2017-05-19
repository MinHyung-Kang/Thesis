function  theta = svgd(theta0, dlog_p_ori, max_iter, opts)
%%%%%%%%
% Bayesian Inference via Stein Variational Gradient Descent
% Implemented from Liu, Q. and Wang, D. (2016) Stein Variational Gradient Descent

% Input:
%   -- theta0: initialization of particles, m * d matrix (m is the number of particles, d is the dimension)
%   -- dlog_p: function handle of first order derivative of log p(x)
%   -- max_iter: maximum iterations
%   -- opts : other options for training

% output:
%   -- theta: a set of particles that approximates p(x)
%%%%%%%%

if ~isfield(opts, 'master_stepsize'); opts.master_stepsize = 0.1; end

% for the following parameters, we always use the default settings
if ~isfield(opts, 'auto_corr'); opts.auto_corr = 0.9; end
if ~isfield(opts, 'method'); opts.method = 'adagrad'; end
if ~isfield(opts, 'kernel_opts')
    opts.kernel_opts.h = -1;
    opts.kernel_opts.approx = 'none';
else
    if ~isfield(opts.kernel_opts,'h'); opts.kernel_opts.h = -1;end
end



switch lower(opts.method)
case 'adagrad'
    %% AdaGrad with momentum
    theta = theta0;

    fudge_factor = 1e-6;
    historical_grad = 0;
    y_historical_grad = 0;
    h_historical_grad = 0;

    if strcmp(opts.kernel_opts.method,'inducedPoints') == 1
        Y = theta(randi([1,size(theta,1)],opts.kernel_opts.m,1),:);
        opts.kernel_opts.Y = Y;
        h = -1;
    end
    opts.currentSeed = opts.baseSeed;
    dlog_p = @(val)dlog_p_ori(val, opts.currentSeed);

    for iter = 1:max_iter

        [grad, ksdInfo] = KSD_KL_gradxy(theta, dlog_p, opts.kernel_opts);   %\Phi(theta)

        [adam_grad,historical_grad] = getAdamUpdate(grad, historical_grad, opts.master_stepsize, opts.auto_corr, fudge_factor);
        theta = theta + adam_grad; % update

        opts.currentseed = opts.currentSeed + 1;
        dlog_p = @(val)dlog_p_ori(val, opts.currentSeed);

        if  opts.kernel_opts.adver > 0
            Y = theta(randi([1,size(theta,1)],opts.kernel_opts.m,1),:);
            opts.kernel_opts.Y = Y;
            h = -1;
            opts.kernel_opts.h = h;
            Sqx = dlog_p(theta);
        end

        for adverInd = 1:opts.kernel_opts.adverIter
            % If using adversairal updates for y
            if opts.kernel_opts.adver == 1 || opts.kernel_opts.adver == 3
                [yGrad, ~] = inducedKernel_grady(theta, Y, Sqx, opts.kernel_opts);
                [adam_yGrad,y_historical_grad] = getAdamUpdate(yGrad, y_historical_grad, opts.master_stepsize, opts.auto_corr, fudge_factor);
                Y = Y + adam_yGrad; % update
                opts.kernel_opts.Y = Y;
            end

            % If using adversairal updates for h
            if opts.kernel_opts.adver == 2 || opts.kernel_opts.adver == 3
                [hGrad, info] = inducedKernel_gradh(theta, Y, Sqx, opts.kernel_opts);
                if h == -1; h = info.bandwidth; end
                [adam_hGrad,h_historical_grad] = getAdamUpdate(hGrad, h_historical_grad, opts.master_stepsize, opts.auto_corr, fudge_factor);
                h = h + adam_hGrad; % update
                opts.kernel_opts.h = h;
            end

        end
    end

otherwise
    error('wrong method');
end
end
