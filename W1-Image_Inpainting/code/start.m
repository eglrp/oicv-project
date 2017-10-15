%Example script: You should replace the beginning of each function ('sol')
%with the name of your group. i.e. if your gropu name is 'G8' you should
%call :
% G8_DualTV_Inpainting_GD(I, mask, paramInp, paramROF)

clearvars;
%There are several black and white images to test:
%  image1_toRestore.jpg
%  image2_toRestore.jpg
%  image3_toRestore.jpg
%  image4_toRestore.jpg
%  image5_toRestore.jpg

available_images = {'image1', 'image2', 'image3', 'image4', 'image5'};

for im_index=1:length(available_images)
    name = available_images{im_index};
    im_path = strcat(name, '_toRestore.jpg');
    im_mask = strcat(name,'_mask.jpg');

    I = double(imread(im_path));
    %I=I(1:10,1:10);

    %Number of pixels for each dimension, and number of channles
    [ni, nj, nC] = size(I);

    if nC==3
        I = mean(I,3); %Convert to b/w. If you load a color image you should comment this line
    end

    %Normalize values into [0,1]
    I=I-min(I(:));
    I=I/max(I(:));

    %Load the mask
    mask_img = double(imread(im_mask));
    %mask_img =mask_img(1:10,1:10);
    [ni, nj, nC] = size(mask_img);
    if nC==3
        mask_img = mask_img(:,:,1); %Convert to b/w. If you load a color image you should comment this line
    end
    %We want to inpaint those areas in which mask == 1
    mask = mask_img >128; %mask(i,j) == 1 means we have lost information in that pixel
                          %mask(i,j) == 0 means we have information in that
                          %pixel

    %%%Parameters for gradient descent (you do not need for week1)
    param.dt = 5*10^-7;
    param.iterMax = 10^4;
    param.tol = 10^-5;

    %%Parameters
    param.hi = 1 / (ni-1);
    param.hj = 1 / (nj-1);

    figure('Name', sprintf('Before inpainting (%s)', im_path), 'NumberTitle','off');
    imshow(I);

    figure('Name', sprintf('Mask (%s)', im_mask), 'NumberTitle','off');
    imshow(mask);

    Iinp=sol_Laplace_Equation_Axb(I, mask, param);
    figure('Name', sprintf('Inpainted (%s)', im_path), 'NumberTitle','off');
    imshow(Iinp);
end


%% Challenge image. (We have lost 99% of information)
clearvars
im_path = 'image6_toRestore.tif';
im_mask = 'image6_mask.tif';
I=double(imread(im_path));
%Normalize values into [0,1]
I=I/256;


%Number of pixels for each dimension, and number of channels
[ni, nj, nC] = size(I);

mask_img=double(imread(im_mask));
mask = mask_img >128; %mask(i,j) == 1 means we have lost information in that pixel
                      %mask(i,j) == 0 means we have information in that
                      %pixel

param.hi = 1 / (ni-1);
param.hj = 1 / (nj-1);

figure('Name', sprintf('Before inpainting (%s)', im_path), 'NumberTitle','off')
imshow(I);

figure('Name', sprintf('Mask (%s)', im_mask), 'NumberTitle','off');
imshow(mask_img);

Iinp(:,:,1)=sol_Laplace_Equation_Axb(I(:,:,1), mask(:,:,1), param);
Iinp(:,:,2)=sol_Laplace_Equation_Axb(I(:,:,2), mask(:,:,2), param);
Iinp(:,:,3)=sol_Laplace_Equation_Axb(I(:,:,3), mask(:,:,3), param);

figure('Name', sprintf('Inpainted (%s)', im_path), 'NumberTitle','off');
imshow(Iinp);

%% Goal Image
clearvars;

im_path = 'Image_to_Restore.png';

%Read the image
I = double(imread(im_path));

[ni, nj, nC] = size(I);


I = I - min(I(:));
I = I / max(I(:));

%We want to inpaint those areas in which mask == 1 (red part of the image)
I_ch1 = I(:,:,1);
I_ch2 = I(:,:,2);
I_ch3 = I(:,:,3);

%TO COMPLETE 1
%mask_img(i,j) == 1 means we have lost information in that pixel
                                      %mask(i,j) == 0 means we have information in that pixel
mask = I_ch1==1 & I_ch2==0 & I_ch3==0;

%%%Parameters for gradient descent (you do not need for week1)
param.dt = 5*10^-7;
param.iterMax = 10^4;
param.tol = 10^-5;

%parameters
param.hi = 1 / (ni-1);
param.hj = 1 / (nj-1);

% for each channel 
figure('Name', sprintf('Before inpainting (%s)', im_path), 'NumberTitle','off')
imshow(I);

figure('Name', 'Computed Mask', 'NumberTitle','off');
imshow(mask);

Iinp(:,:,1)=sol_Laplace_Equation_Axb(I_ch1, mask, param);
Iinp(:,:,2)=sol_Laplace_Equation_Axb(I_ch2, mask, param);
Iinp(:,:,3)=sol_Laplace_Equation_Axb(I_ch3, mask, param);
figure('Name', sprintf('Inpainted (%s)', im_path), 'NumberTitle','off');
imshow(Iinp);