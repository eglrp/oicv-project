function [ result ] = sol_DjCentral(  I, hj  )
%SOL_DJCENTRAL Summary of this function goes here
%   Detailed explanation goes here
    if (~exist('hj', 'var'))
        hj=1;
    end

    result=I;
    %Begin To Complete 11
    result(:, 2:end-1) = (I(:, 3:end)-I(:, 1:end-2))./(2*hj); %result(:, 2:end)
    %End To Complete 11

end

