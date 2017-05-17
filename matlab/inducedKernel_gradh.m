function [hGrad, info] = KSD_inducedKernel_gradh(x, y, Sqx, kernel_opts)
%%%%%%%%%%%%%%%%%%%%%%
% Returns gradient of KSD of induced kernel with respect to h
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

%% Preprocessing stepss
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

if h == -1
    h = getMedian(H);  % Median trick
end
h2 = h^2;

Kxy = exp(-H/(2*h2)); % Calculate rbf kernel

hGrad = 0;

% Get gradient with respect to h
for yInd = 1:m
    Kxy_cur = Kxy(:,yInd);
    H_cur = H(:,yInd);
    xmy = x-repmat(y(yInd,:),n,1);
    Sqxxmy = Sqx - xmy/h2;

    part2 = repmat(Kxy_cur,1,d) .* Sqxxmy;
    part1 = repmat(H_cur/h^3,1,d) .* part2 + repmat(Kxy_cur,1,d) .* (2*xmy/h^3);

    part = part1 * part2';
    hGrad = hGrad + sum(sum(part,2),1);

    % For U_Statistic (subtract the case of diagonal, when x=x')
    if kernel_opts.uStat
        front_u = Kxy_cur.^2 .* H_cur/h^3 .* sum(Sqxxmy.^2,2);
        back_u = sum(2*xmy/h^3 .* Sqxxmy,2);
        hGrad = hGrad - sum(Kxy_cur.^2 .* (front_u + back_u),1);
    end
end

if kernel_opts.uStat
    hGrad = hGrad * 2/ (n*(n-1)*m);
else
    hGrad = hGrad * 2/ (n^2*m);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

info.bandwidth = h;

return;


end
