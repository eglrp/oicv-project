function [ phi, c1, c2 ] = G8_ChanVeseIpol_GDExp( I, phi_0, mu, nu, eta, lambda1, ...
    lambda2, tol, epHeaviside, dt, iterMax, reIni, fname, plot_iters)
    %Implementation of the Chan-Vese segmentation following the explicit
    %gradient descent in the paper of Pascal Getreur "Chan-Vese Segmentation".
    %It is the equation 19 from that paper

    %I     : Gray color image to segment
    %phi_0 : Initial phi
    %mu    : mu lenght parameter (regularizer term)
    %nu    : nu area parameter (regularizer term)
    %eta   : epsilon for the total variation regularization
    %lambda1, lambda2: data fidelity parameters
    %tol   : tolerance for the sopping criterium
    % epHeaviside: epsilon for the regularized heaviside. 
    % dt     : time step
    %iterMax : MAximum number of iterations
    %reIni   : Iterations for reinitialization. 0 means no reinitializacion
    %fname   : Name of the file, used to save predictions / curve evolution
    %plot_iters : Number of iterations between plots

    [ni,nj]=size(I);
    hi=1;
    hj=1;


    phi=phi_0;
    dif=inf;
    nIter=0;
    while dif>tol && nIter<iterMax

        phi_old=phi;
        nIter=nIter+1;        


        %Fixed phi, Minimization w.r.t c1 and c2 (constant estimation)
        %Clarification: H(phi) = 1 when phi>=0 and =0 when phi<0
        c1 = sum(sum(I(phi_old>=0)))./nnz(phi_old>=0); %EQ 10 of IPOL_Getreuer TODO 1: Line to complete
        c2 = sum(sum(I(phi_old<0)))./nnz(phi_old<0); %EQ 11 of IPOL_Getreuer TODO 2: Line to complete

        %Boundary conditions --> page 5 ofIPOL_Getreuer_...
        phi(1,:)   = phi(2,:); %TODO 3: Line to complete
        phi(end,:) = phi(end-1,:); %TODO 4: Line to complete

        phi(:,1)   = phi(:,2); %TODO 5: Line to complete
        phi(:,end) = phi(:, end-1); %TODO 6: Line to complete


        %Regularized Dirac's Delta computation
        delta_phi = G8_diracReg(phi, epHeaviside);   %notice delta_phi=H'(phi)	

        %derivatives estimation
        %i direction, forward finite differences
        phi_iFwd  = DiFwd(I, hi); %TODO 7: Line to complete
        phi_iBwd  = DiBwd(I, hi); %TODO 8: Line to complete

        %j direction, forward finitie differences
        phi_jFwd  = DjFwd(I, hj); %TODO 9: Line to complete
        phi_jBwd  = DjBwd(I, hj); %TODO 10: Line to complete

        %centered finite diferences
        %phi_icent   = DiCent(I, hi); %TODO 11: Line to complete
        %phi_jcent   = DjCent(I, hj); %TODO 12: Line to complete
        phi_icent = (phi_iFwd + phi_iBwd)/2;
        phi_jcent = (phi_jFwd + phi_jBwd)/2;

        %A and B estimation (A y B from the Pascal Getreuer's IPOL paper "Chan
        %Vese segmentation
        A = mu./sqrt(eta.^2 + phi_iFwd.^2 +phi_jcent.^2);   %eq 18   TODO 13: Line to complete
        B = mu./sqrt(eta.^2 + phi_icent.^2 + phi_jFwd.^2);  %eq 18   TODO 14: Line to complete


        %%Equation 22, for inner points
        phi(2:end-1, 2:end-1) = (phi_old(2:end-1, 2:end-1) + dt*delta_phi(2:end-1,2:end-1).*...
            (A(2:end-1, 2:end-1).*phi_old(3:end,2:end-1)+ A(1:end-2, 2:end-1).*phi_old(1:end-2,2:end-1)+...
            B(2:end-1, 2:end-1).*phi_old(2:end-1, 3:end)+B(2:end-1, 1:end-2).*phi_old(2:end-1,1:end-2)...
            - nu - lambda1*(I(2:end-1,2:end-1) - c1).^2 + lambda2*(I(2:end-1,2:end-1) - c2).^2 ) ) ...
            ./(1 + dt*delta_phi(2:end-1,2:end-1).*(A(2:end-1,2:end-1) + A(1:end-2,2:end-1) +...
            B(2:end-1, 2:end-1)+ B(2:end-1, 1:end-2))); %TODO 15: Line to complete

        %Reinitialization of phi
        if 	reIni>0 && mod(nIter, reIni)==0
            indGT = phi >= 0;
            indLT = phi < 0;

            phi=double(bwdist(indLT) - bwdist(indGT));

            %Normalization [-1 1]
            nor = min(abs(min(phi(:))), max(phi(:)));
            phi=phi/nor;
        end

        %Diference. This stopping criterium has the problem that phi can
        %change, but not the zero level set, that it really is what we are
        %looking for.
        dif = mean(sum( (phi(:) - phi_old(:)).^2 ));
        
        if mod(nIter-1, plot_iters) == 0
            
            fig = figure('doublebuffer','off','Visible','Off');
            %Plot the level sets surface
            ax1 = subplot(1,2,1);
            %The level set function
            hold on;
            surfc(phi);
            view(25, 20);  % Surface viewpoint
            contour(phi>0); %The zero level set over the surface
            hold off;
            colormap(ax1, 'hot');
            title('Phi Function');

            %Plot the curve evolution over the image
            ax2 = subplot(1,2,2);
            imagesc(I);        
            hold on;
            contour(phi>0, 'Color', 'r');
            axis off;
            hold off;
            colormap(ax2, 'gray');
            title('Image and zero level set of Phi');
            
            % Save plot
            [~, filename, ~] = fileparts(fname);
            save_path = fullfile('code', 'curve_evolution', filename, sprintf('iter%d.png', nIter));
            saveas(fig, save_path);
        end
        
        % Exponential growth of time interval between captured frames
        % Every 10 frames we duplicate the time interval (plot_iters)
        if mod(nIter, 10*plot_iters)==0
            plot_iters=plot_iters*2;
        end
        
        fprintf('Iter: %d\n', nIter);
        fprintf('Diff: %s\n', dif);
        fprintf('\n');
    end
    fig = figure('doublebuffer','off','Visible','Off');
    %Plot the level sets surface
    ax1 = subplot(1,2,1);
    %The level set function
    hold on;
    surfc(phi);
    view(25, 20);  % Surface viewpoint
    contour(phi>0); %The zero level set over the surface
    hold off;
    colormap(ax1, 'hot');
    title('Phi Function');

    %Plot the curve evolution over the image
    ax2 = subplot(1,2,2);
    imagesc(I);        
    hold on;
    contour(phi>0, 'Color', 'r');
    axis off;
    hold off;
    colormap(ax2, 'gray');
    title('Image and zero level set of Phi');

    % Save plot
    [~, filename, ~] = fileparts(fname);
    save_path = fullfile('code', 'curve_evolution', filename, sprintf('iter%d.png', nIter));
    saveas(fig, save_path);
end