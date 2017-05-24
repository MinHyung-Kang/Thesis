function [MSE_x, MSE_xsq] = evalPoints(points, x_nuts, cov_nuts)
    x_hat = mean(points,1);
    cov_hat = cov(points);
    
    MSE = @(y,yhat)(mean((y(:) - yhat(:)).^2));

    MSE_x = log10(MSE(x_hat, x_nuts));
    MSE_xsq = log10(MSE(cov_hat, cov_nuts));
end