clear all;
close all;
clc;

im_name='7_9_s.bmp';

% TODO: Update library path
% Add  library paths
% basedir='~/Desenvolupament/UGM/';
% addpath(basedir);

im_names = {'7_9_s.bmp', '2_1_s.bmp', '3_12_s.bmp'};


%Set model parameters
%cluster color
% K=20; % Number of color clusters (=number of states of hidden variables)

%Pair-wise parameters
ii=-1;
smooth_term=[0.0 10]; % Potts Model
for i = 1:length(im_names)
    ii=ii+1;
    for K = 4:5:20
        ii=ii+1;
        %Load images
        im = imread(im_names{i});


        NumFils = size(im,1);
        NumCols = size(im,2);

        %Convert to LAB colors space
        % TODO: Uncomment if you want to work in the LAB space
        %
        im = RGB2Lab(im);

        %Preparing data for GMM fiting
        %
        % TODO: define the unary energy term: data_term
        % nodePot = P( color at pixel 'x' | Cluster color 'c' )  

        im=double(im);
        x=reshape(im, [size(im,1)*size(im,2) size(im,3)]);
        gmm_color = gmdistribution.fit(x,K);
        mu_color = gmm_color.mu;

        data_term=gmm_color.posterior(x);
        nodePot = data_term;

        %Building 4-grid
        %Build UGM Model for 4-connected segmentation
        disp('create UGM model');

        % Create UGM data
        [edgePot,edgeStruct] = CreateGridUGMModel(NumFils, NumCols, K ,smooth_term);


        if ~isempty(edgePot)

            % color clustering
            [~,c] = min(reshape(data_term,[NumFils*NumCols K]),[],2);
            im_c= reshape(mu_color(c,:),size(im));

        %     Call different UGM inference algorithms
            display('Loopy Belief Propagation'); tic;
            [nodeBelLBP,edgeBelLBP,logZLBP] = UGM_Infer_LBP(nodePot,edgePot,edgeStruct);toc;
            [ value_max , index_max] = (max(nodeBelLBP, [], 2));
            im_lbp = reshape(mu_color(index_max,:), size(im));

            % Max-sum
            display('Max-sum'); tic;
            decodeLBP = UGM_Decode_LBP(nodePot,edgePot,edgeStruct);
            im_bp= reshape(mu_color(decodeLBP,:),size(im));
            toc;


            % TODO: apply other inference algorithms and compare their performance
            %
            % - Graph Cut
        %     display('Graph cut'); tic;
        %     decodeGcut = UGM_Decode_GraphCut(nodePot,edgePot,edgeStruct);
        %     im_gcut = reshape(mu_color(decodeGcut,:), size(im));
        %     toc;

            % - Linear Programing Relaxation
        %     display('Linear Programming'); tic;
        %     decodeLinProg = UGM_Decode_LinProg(nodePot,edgePot,edgeStruct);
        %     im_linprog = reshape(mu_color(decodeLinProg, :), size(im));
        %     toc;

            figure (ii)
            subplot(2,2,1),imshow(Lab2RGB(im));xlabel('Original');
            subplot(2,2,2),imshow(Lab2RGB(im_c),[]);xlabel('Clustering without GM');
            subplot(2,2,3),imshow(Lab2RGB(im_bp),[]);xlabel('Max-Sum');
            subplot(2,2,4),imshow(Lab2RGB(im_lbp),[]);xlabel('Loopy Belief Propagation');
            %subplot(2,2,4),imshow(Lab2RGB(im_linprog),[]);xlabel('Linear Programming');
            print(fullfile(sprintf('K_%s_%s',string(K),im_names{i})), '-dpng' );
        else

            error('You have to implement the CreateGridUGMModel.m function');

        end
    end
end