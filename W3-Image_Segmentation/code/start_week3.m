%close all;
clearvars;
clc

%% Parameters
names = {'circles.png', 'noisedCircles.tif', 'phantom17.bmp', 'phantom18.bmp', 'phantom19.bmp', 'Image_to_Restore.png'};

mus = [1, 1, 1, 0.5, 10, 10];
nu=0;
lambdas = [1, 1, 1, 1, 1, 10^-3];

epHeaviside=1;
eta=1;
tols=[0.09, 1e-3, 0.1, 0.1, 0.01, 0.001];

dts=(10^-1)./mus;
iterMax=5000;
reIni=1500;

plot_iters = 10;

% Length and area parameters
    % circles.png mu=1, mu=2, mu=10
    % noisedCircles.tif mu=0.1
    % phantom17 mu=1, mu=2, mu=10
    % phantom18 mu=0.2 mu=0.5
    % hola carola mu=1
% Other parameters  
    % eta=0.01;
    % dt=(10^-2)/mu; 
    
%% Main cycle
iters = 1:length(names);  % Change this value if you want to segment a subset of the images

for i=iters
    fname = names{i};
    fprintf('Processing file %s\n', fname);
    I=double(imread(fname));
    I=mean(I,3);
    I=I-min(I(:));
    I=I/max(I(:));
    
    [ni, nj]=size(I);

    mu=mus(i);
    lambda1=lambdas(i);
    lambda2=lambdas(i);
    dt=dts(i);
    tol=tols(i);
    [X, Y]=meshgrid(1:nj, 1:ni);

    %%Initial phi
    if strcmp(fname, 'Image_to_Restore.png') || strcmp(fname, 'phantom19.bmp')
        phi_0=I;
    elseif strcmp(fname, 'phantom18.bmp') || strcmp(fname, 'phantom17.bmp')
        phi_0=(-sqrt( ( X-round(ni/2)).^2 + (Y-round(nj/4)).^2)+50);
    else
        phi_0=(-sqrt( ( X-round(ni/2)).^2 + (Y-round(nj/2)).^2)+50);
    end
    phi_0=phi_0-min(phi_0(:));
    phi_0=2*phi_0/max(phi_0(:));
    phi_0=phi_0-1;


    %%Explicit Gradient Descent
    seg=G8_ChanVeseIpol_GDExp( I, phi_0, mu, nu, eta, lambda1, lambda2, tol, epHeaviside, dt, iterMax, reIni, fname, plot_iters);
end
