function [Akxy, info] = KSD_KL_gradxy(x, dlog_p, kernel_opts)
%%%%%%%%%%%%%%%%%%%%%%
% Returns update for SVGD particles
% Implemented from Liu, Q. and Wang, D. (2016) Stein Variational Gradient Descent
%
% Input:
%    -- x: particles, n*d matrix, where n is the number of particles and d is the dimension of x
%    -- dlog_p: a function handle, which returns the first order derivative of log p(x), n*d matrix
%    -- kernel_opts : more options
%       - kernel_opts.h = bandwidth. If h == -1, h is selected by the median trick
%       - kernel_opts.method = name of the method
%           ['none','subset','subsetCF','inducedPoints']
%       - kernel_opts.m = size of subparticles to use
%       - kernel_opts.Y = if using subparticles

% Output:
%    --Akxy: n*d matrix, \Phi(x) is our algorithm, which is a smooth
%    function that characterizes the perturbation direction
%    --info: kernel bandwidth
%%%%%%%%%%%%%%%%%%%%%%
[n, d] = size(x);

h = kernel_opts.h;
if strcmp(kernel_opts.method,'none') == 0
    m = kernel_opts.m;
end

%%%%%%%%%%%%%% Main part %%%%%%%%%%
x2 = sum(x.^2,2);
method = kernel_opts.method;

getMedian = @(mat)(sqrt(0.5*median(mat(:)) / log(n+1)));

if strcmp(method,'none') % Induced Kernel Method
    Sqx = dlog_p(x);
    % Using rbf kernel as default
    XY = x*x';
    X2e = repmat(x2, 1, n);

    H = (X2e + X2e' - 2*XY); % Calculate pairwise distance
    % median trick for bandwidth
    if h == -1
        h = getMedian(H);    %rbf_dot has factor two in kernel
    end
    h2 = h^2;
    Kxy = exp(-H/(2*h2));   % calculate rbf kernel

    dxKxy= -Kxy * x;
    sumKxy = sum(Kxy,2);

    dxKxy = (dxKxy + x .* repmat(sumKxy,1,d)) / h2;
    Akxy = (Kxy * Sqx  + dxKxy)/n;

elseif ~isempty(strfind(method,'subset')) % Subset method
    mInd  = randsample(n,m);
    y = x(mInd,:);
    y2 = x2(mInd);

    XY = x*y';
    X2e = repmat(x2, 1, m);
    Y2e = repmat(y2',n,1);
    H = (X2e + Y2e - 2*XY);

    % median trick for bandwidth
    if h == -1
        h = getMedian(H);  %rbf_dot has factor two in kernel
    end
    h2 = h^2;
    Kxy = exp(-H/(2*h2)); % Calculate rbf kernel
    Sqy = dlog_p(y);

    % Peform control functional if specified
    if ~isempty(strfind(method,'CF'))
        sqxdy = -(Sqy*y' - repmat(sum((Sqy.*y),2),1,m))./h2;
        dxsqy = sqxdy';
        dxdy = (-H(mInd,:)/(h2^2) + d/h2);
        KxySub = Kxy(mInd,:);

        KpMat = (Sqy*Sqy' + sqxdy + dxsqy + dxdy).*KxySub;

        weights = getWeights(KpMat);
        info.w = weights;
        info.y = y;

        Kxy = Kxy .* repmat(weights,n,1);
    end

    dxKxy= -Kxy * y;
    sumKxy = sum(Kxy,2);
    dxKxy = (dxKxy + x .* repmat(sumKxy,1,d)) / h2;
    Akxy = (Kxy * Sqy  + dxKxy);

    if ~isempty(strfind(method,'CF'))
        Akxy = Akxy / m;
    end

elseif ~isempty(strfind(method,'inducedPoints'))% Induced Points method
    if isfield(kernel_opts, 'Y')
        y = kernel_opts.Y;
        y2 = sum(y.^2, 2);
    else
        mInd  = randsample(n,m);
        y = x(mInd,:);
        y2 = x2(mInd);
    end

    XY = x*y';
    X2e = repmat(x2, 1, m);
    Y2e = repmat(y2',n,1);
    H = (X2e + Y2e - 2*XY);

    if h == -1
        h = getMedian(H);  %rbf_dot has factor two in kernel
    end
    h2 = h^2;

    Kxy = exp(-H/(2*h2)); % Calculate rbf kernel
    Sqx = dlog_p(x);

    innerTerm = (Kxy' * (Sqx - x/h2) + repmat(sum(Kxy,1)',1,d) .* y/h2)/n;

    Akxy = (Kxy * innerTerm)/m;
    
    info.Sqx = Sqx;
else
    error('Unrecognized method');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

info.bandwidth = h;
return;


function [weights] = getWeights(kpmat)
    lambda = getConditionNumber(kpmat);
    z = size(kpmat,1);

    % Get weights
    KPrime = kpmat + lambda * z * eye(z);
    num = ones(1,z) / KPrime;
    denom = 1 + ones(1,z) / KPrime * ones(z,1);
    weights = num ./ denom;

    % Normalize weights? but they are negative and don't add up to 1
    weights = weights ./ sum(weights,2);
    %weights = abs(weights) ./ sum(abs(weights),2);
end

% Given a kernel matrix K, let lambda be smallest power of 10 such that
% kernel matrix K0 + lamba*I has condition number lower than 10^10
% Note we use 2-norm for computing condition number
function [lambda] = getConditionNumber(K)
    lambda = 10e-10;
    condA = 10e11;
    matSize = size(K,1);
    while (condA > 10e10)
        lambda = lambda * 10;
        A = K + lambda * eye(matSize);
        condA = norm(A) * norm(inv(A));
    end

end
end
