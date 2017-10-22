function [ result ] = sol_DiCentral(  I, hi  )
%SOL_DICENTRAL Summary of this function goes here
%   Detailed explanation goes here
    if (~exist('hi', 'var'))
        hi=1;
    end

    result=I;
    %Begin To Complete 11
    result(2:end-1, :) = (I(3:end, :)-I(1:end-2, :))./(2*hi); 
    %result = (sol_DiBwd(I, hi) + sol_DiFwd(I, hi))/2;
    %End To Complete 11

end


