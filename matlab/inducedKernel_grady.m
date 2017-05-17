function [yGrad, info] = inducedKernel_grady(x, y, Sqx, kernel_opts)
%%%%%%%%%%%%%%%%%%%%%%
% Returns gradient of KSD of induced kernel with respect to y
%
% Input:
%    -- x: particles, n*d matrix, where n is the number of particles and d is the dimension of x
%    -- y: subparticles with which we want to take gradient of induced KSD
%    -- Sqx : result of dlog_p(x)
%    -- kernel_opts : more options
%       - kernel_opts.h = bandwidth. If h == -1, h is selected by the median trick
%       - kernel_opts.method = name of the method
%           ['none','subset','subsetCF','inducedPoints']
%       - kernel_opts.m = size of subparticles to use
%       - kernel_opts.Y = if using subparticles

% Output:
%    --yGrad : m*d matrix, amount to change for y
%    --info : additional information
%%%%%%%%%%%%%%%%%%%%%%


% Preprocessing stepss
[n, d] = size(x);

m = kernel_opts.m;
h = kernel_opts.h;

% Function handle for median trick
getMedian = @(mat)(sqrt(0.5*median(mat(:)) / log(n+1)));

%%%%%%%%%%%%%% Main part %%%%%%%%%%
x2= sum(x.^2, 2);
y2 =sum(y.^2, 2);

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

yGrad = zeros(m,d);

% Compute gradient
for yInd = 1:m
    Kxy_cur = Kxy(:,yInd);
    xmy = (x - repmat(y(yInd,:),n,1))/h^2;
    Sqxxmy = (Sqx - xmy);
    back = repmat(Kxy_cur,1,d) .* Sqxxmy;
    yGrad(yInd,:) = sum(xmy .* repmat(sum(back*back',2),1,d),1) + ...
                        sum(back,1) * sum(Kxy_cur)/h^2;

    % For U_Statistic (subtract the case of diagonal, when x=x')
    if kernel_opts.uStat
        front_u = repmat( (Kxy_cur.^2) .* sum(Sqxxmy.^2,2),1,d) .* xmy;
        back_u = repmat((Kxy_cur.^2)/h2,1,d) .* Sqxxmy; % n by d

        yGrad(yInd,:) = yGrad(yInd,:) - sum(front_u + back_u,1);
    end

end

% If using Ustat or not
if kernel_opts.uStat
    yGrad = yGrad * 2 / (n*(n-1) * m);
else
    yGrad = yGrad * 2 / (n^2 * m);
end



% Regularization
if kernel_opts.alpha > 0
    Y2e = repmat(y2, 1, m);
    H_y = (Y2e + Y2e' - 2*(y*y'));
    Kxy_y = exp(-H_y/(2*h2));
    sumKxy_y = sum(Kxy_y,2);
    yReg = (y .* repmat(sumKxy_y,1,d) - Kxy_y * y) / (h2 * m);

    yGrad = yGrad + kernel_opts.alpha  * yReg;
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

info.bandwidth = h;

return;
