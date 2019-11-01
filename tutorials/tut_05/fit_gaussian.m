function [m, S] = fit_gaussian(X)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
    m = mean(X);
    S = cov(X);
end

