clearvars;
dst = double(imread('lena.png'));
src = double(imread('girl.png')); % flipped girl, because of the eyes
[ni,nj, nChannels]=size(dst);

param.hi=1;
param.hj=1;

% How to compute the laplacian of the inserted image.
% Options:
%   * 'forward': use the forward derivative to compute the gradient from
%      the image and then the laplacian from the gradient
%
%   * 'backward': use the backward derivative to compute the gradient from
%      the image and then the laplacian from the gradient
% 
%   * 'finite_differences': apply the finite differences formula directly
%      to the image to compute the laplacian.
param.laplacian_method = 'forward';  


%masks to exchange: Eyes
mask_src=logical(imread('mask_src_eyes.png'));
mask_dst=logical(imread('mask_dst_eyes.png'));

for nC = 1: nChannels
    
    % Compute laplacian according to param.laplacian_method
    switch param.laplacian_method
        case 'forward'
            drivingGrad_i = sol_DiFwd(src(:, :, nC), param.hi);
            drivingGrad_j = sol_DjFwd(src(:, :, nC), param.hj);

            driving_on_src = (sol_DiFwd(drivingGrad_i, param.hi)) + ...
                             (sol_DjFwd(drivingGrad_j, param.hj));
        case 'backward'
            drivingGrad_i = sol_DiBwd(src(:, :, nC), param.hi);
            drivingGrad_j = sol_DjBwd(src(:, :, nC), param.hj);

            driving_on_src = (sol_DiBwd(drivingGrad_i, param.hi)) + ...
                             (sol_DjBwd(drivingGrad_j, param.hj));
        case 'finite_differences'
            error('finite differences method not implemented');
        otherwise
            error('param.laplacian_method not one of: forward, backward, finite_differences');
    end
    driving_on_dst = zeros(size(src(:,:,1)));   
    driving_on_dst(mask_dst(:)) = driving_on_src(mask_src(:));
    
    param.driving = driving_on_dst;

    dst1(:,:,nC) = G8_Poisson_Equation_Axb(dst(:,:,nC), mask_dst,  param);
end

%Mouth
%masks to exchange: Mouth
mask_src=logical(imread('mask_src_mouth.png'));
mask_dst=logical(imread('mask_dst_mouth.png'));
for nC = 1: nChannels
    
    % Compute laplacian according to param.laplacian_method
    switch param.laplacian_method
        case 'forward'
            drivingGrad_i = sol_DiFwd(src(:, :, nC), param.hi);
            drivingGrad_j = sol_DjFwd(src(:, :, nC), param.hj);

            driving_on_src = (sol_DiFwd(drivingGrad_i, param.hi)) + ...
                             (sol_DjFwd(drivingGrad_j, param.hj));
        case 'backward'
            drivingGrad_i = sol_DiBwd(src(:, :, nC), param.hi);
            drivingGrad_j = sol_DjBwd(src(:, :, nC), param.hj);

            driving_on_src = (sol_DiBwd(drivingGrad_i, param.hi)) + ...
                             (sol_DjBwd(drivingGrad_j, param.hj));
        case 'finite_differences'
            % TODO: Implement finite differences method to compute
            % Laplacian, that is, use the formula directly
            error('finite differences method not implemented');
        otherwise
            error('param.laplacian_method not one of: forward, backward, finite_differences');
    end
    
    driving_on_dst = zeros(size(src(:,:,1)));  
    driving_on_dst(mask_dst(:)) = driving_on_src(mask_src(:));
    
    param.driving = driving_on_dst;

    dst1(:,:,nC) = G8_Poisson_Equation_Axb(dst1(:,:,nC), mask_dst,  param);
end

imshow(dst1/256)