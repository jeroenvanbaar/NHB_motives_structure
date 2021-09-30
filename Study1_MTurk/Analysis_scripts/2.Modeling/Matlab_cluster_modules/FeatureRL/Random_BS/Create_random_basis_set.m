%% Create random basis set

function basis = Create_random_basis_set(varargin)
    
    %% Parse variable arguments
    visualize = false;
    for varg = 1:length(varargin)
        if ischar(varargin{varg})
            if strcmpi('visualize',varargin{varg})
                visualize = varargin{varg + 1};
                varargin{varg} = {}; varargin{varg + 1} = {};
            end
        end
    end
    
    %% Initialize grid and pick random initial point
    grid = zeros(4,4);
    start_Ti = randi(4);
    start_Si = randi(4);
    grid(start_Si, start_Ti) = 1;
    
    if visualize
        fig = figure; %#ok<NASGU>
        img_handle = imagesc(grid);
        caxis([0,1]);
        axis equal
        axis tight;
        colorbar;
    end
    
    %% Grow
    strel_use = [0,1,0;1,1,1;0,1,0];
%     grids_store = grid;
    while sum(sum(grid)) < 8
        
        boundary_elements = find(imdilate(grid,strel_use) - grid);
        grid(boundary_elements(randi(length(boundary_elements)))) = 1;
        
        if visualize
            set(img_handle, 'CData', grid);
            pause(.5);
        end
        
%         grids_store(:,:,end+1) = grid;
%         pause(.5);
    end
    
    %% Return
    basis = grid;
    
end