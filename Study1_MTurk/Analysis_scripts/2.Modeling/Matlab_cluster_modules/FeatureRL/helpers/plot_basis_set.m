function plot_basis_set(basis_set, varargin)
    
    % Parse arguments
    figure_handle = [];
    custom_titles = {};
    nrows = 1;
    for varg = 1:length(varargin)
        if ischar(varargin{varg})
            if strcmpi('figure_handle',varargin{varg})
                figure_handle = varargin{varg + 1};
                varargin{varg} = {}; varargin{varg + 1} = {};
            end
            if strcmpi('custom_titles',varargin{varg})
                titles = varargin{varg + 1};
                varargin{varg} = {}; varargin{varg + 1} = {};
            end
            if strcmpi('nrows',varargin{varg})
                nrows = varargin{varg + 1};
                varargin{varg} = {}; varargin{varg + 1} = {};
            end
        end
    end
    if isempty(figure_handle)
        figure_handle = figure();
    end
    
    % Cmap
    hex = flipud(['#B2182B'; '#D6604D'; '#F4A582'; '#FDDBC7'; '#FFFFFF'; '#D1E5F0'; '#92C5DE'; '#4393C3'; '#2166AC']);
    vec = (100:-(100/(length(hex)-1)):0)';
    raw = sscanf(hex','#%2x%2x%2x',[3,size(hex,1)]).' / 255;
    N = 128;
    %N = size(get(gcf,'colormap'),1) % size of the current colormap
    cmap = interp1(vec,raw,linspace(100,0,N),'pchip');
%     cmap = colormap('parula');
    
    % Plot
    figure(figure_handle);
    colormap(cmap);
    n_bases = length(basis_set);
    
%     grids = nan(4,4,n_bases);
    Ss = [10,7,3,0];
    Ts = [5,8,12,15];
    for i = 1:n_bases
        subplot(nrows,ceil(n_bases/nrows),i);
        
        grid = zeros(4,4);
        for x = 1:4
            for y = 1:4
                grid(y,x) = basis_set(i).model(Ss(y),Ts(x));
            end
        end
        imagesc(grid);
        if ~isempty(custom_titles)
            title(titles{i});
        else
            title(basis_set(i).name);
        end
        caxis([-1,1]);
        axis equal;
        axis tight;
%         title(sprintf('Prior weight = %.2f',weights(i)));
        xticks([1,2,3,4]); yticks([1,2,3,4]);
        xticklabels({'5','8','12','15'});
        yticklabels({'10','7','3','0'});
        xlabel('T'); ylabel('S');
    end
    
end