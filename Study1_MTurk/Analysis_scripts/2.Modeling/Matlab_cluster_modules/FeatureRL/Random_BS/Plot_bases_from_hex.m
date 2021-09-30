function weighted_mean = Plot_bases_from_hex(basisSet_hex, weights, plot_bool, figure_handle)
    
    % Cmap
    hex = flipud(['#B2182B'; '#D6604D'; '#F4A582'; '#FDDBC7'; '#FFFFFF'; '#D1E5F0'; '#92C5DE'; '#4393C3'; '#2166AC']);
    vec = (100:-(100/(length(hex)-1)):0)';
    raw = sscanf(hex','#%2x%2x%2x',[3,size(hex,1)]).' / 255;
    N = 128;
    %N = size(get(gcf,'colormap'),1) % size of the current colormap
    cmap = interp1(vec,raw,linspace(100,0,N),'pchip');
    
    n_bases = length(basisSet_hex);
    grids = nan(4,4,n_bases);
    for i = 1:n_bases
        basis_hex = basisSet_hex{i};
        basis_hex_parsed = strsplit(basis_hex,'_');
        ones = basis_hex_parsed{1};
        pointfives = basis_hex_parsed{2};
        grid = zeros(4,4);
        for pos = 1:length(ones)
            index = hex2dec(ones(pos)) + 1;
            grid(index) = 1;
        end
        if ~isempty(pointfives)
            for pos = 1:length(pointfives)
                index = hex2dec(pointfives(pos)) + 1;
                grid(index) = .5;
            end
        end
        grids(:,:,i) = grid;
    end

    if plot_bool
        figure(figure_handle);
        colormap(cmap);
        for i = 1:length(basisSet_hex)
            subplot(1,5,i);
            imagesc(flipud(grids(:,:,i)));
            caxis([0,1]);
            axis equal;
            axis tight;
            title(sprintf('wt=%.2f',weights(i)));
            xticks([1,2,3,4]);
            xticklabels({'5','8','12','15'});
            yticks([1,2,3,4]);
            yticklabels({'10','7','3','0'});
            xlabel('T'); ylabel('S');
%             axis ij;
%             set(gca,'Ydir','reverse');
        end
    end
    
    grids_weighted = nan(4,4,n_bases);
    for i = 1:n_bases
        grids_weighted(:,:,i) = grids(:,:,i).*weights(i);
    end
    weighted_mean = mean(grids_weighted,3);
    
    if plot_bool
        subplot(1,5,5);
        imagesc(flipud(weighted_mean));
%         caxis([-1,1]);
%         caxis(repmat(ceil(max(abs(caxis))*10)/10,[1,2]).*[-1,1]);
        axis equal;
        axis tight;
%         set(gca,'Ydir','reverse');
%         colorbar;
        title('wt-mean');
        xticks([1,2,3,4]);
        xticklabels({'5','8','12','15'});
        yticks([1,2,3,4]);
        yticklabels({'10','7','3','0'});
        xlabel('T'); ylabel('S');
    end
    
end