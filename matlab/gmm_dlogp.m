% Creates a function handle that returns gradp for gmm
function gradp = gmm_dlogp(pdistrib, X, pVal)
    mu = pdistrib.mu;
    sigma = pdistrib.Sigma;
    numComp = pdistrib.NumComponents;
    w = pdistrib.ComponentProportion;
    [N,T] = size(X);
    gradp = zeros(N, T);
    for k = 1:numComp % For each component
        mu_k = mu(k,:);
        covMat_i = sigma(:,:,k);
        w_k = w(k);
        front = mvnpdf(X, mu_k, covMat_i);   % n by 1
        back = (bsxfun(@minus, X, mu(k,:)) * (inv(covMat_i))'); % n by d
        gradp = gradp + (-1) * w_k * bsxfun(@times, front, back);
    end

    if nargin == 2
        pVal = log(pdf(pdistrib,X));
    end

    gradp = bsxfun(@rdivide, gradp, pVal);
end
