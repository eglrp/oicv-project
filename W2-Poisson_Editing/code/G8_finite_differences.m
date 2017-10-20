function [ result ] = G8_finite_differences( I, param )
%G8_FINITE_DIFERENCES Summary of this function goes here
%   Detailed explanation goes here
    [ni, nj] = size(I);
    I_ext = zeros(ni+2, nj+2);
    %here we fill the inner points of I_ext with the values of I
    I_ext(2:end-1, 2:end-1) = I;
    %Here we duplicate the values on the ghost boundaries
    I_ext(2:end-1,1)=I(:,1);
    I_ext(2:end-1,nj+2) = I(:,nj);
    I_ext(1,2:end-1) = I(1,:);
    I_ext(ni+2, 2:end-1) = I(ni,:);
    
    %Here we duplicates the values of the 4 corners
    I_ext(1,1)=I(1,1);
    I_ext(1,nj+2)=I(1,nj);
    I_ext(ni+2,1)=I(ni,1);
    I_ext(ni+2,nj+2)=I(ni,nj);
    
    hi = param.hi;
    hj=param.hj;
    result_ext = zeros(ni+2, nj+2);
    for i = 2:ni+1
        for j=2:nj+1
            result_ext(i,j)= -(2*hi^2 + 2*hj^2)*I_ext(i,j)+hj^2*I_ext(i-1,j)+hj^2*I_ext(i+1,j)+hi^2*I_ext(i,j-1)+hi^2*I_ext(i,j+1);
        end
    end
    result=full(result_ext(2:end-1, 2:end-1));
end

