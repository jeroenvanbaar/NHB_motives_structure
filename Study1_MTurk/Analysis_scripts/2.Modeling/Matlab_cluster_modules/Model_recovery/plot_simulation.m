
function plot_simulation(sim_dat)
% % Load simulations & params
% load('simulations.mat');
% load('params.mat');
% 
% %% Pick simulated model
% model_feature_sets = {'CoST','CoHgShSgPd','CoGrRiNa'};
% smi = 2;
% 
% % Select simulated data
% sim_index = 1;
% simulated_model_feature_set = model_feature_sets{smi};
% sim_dat = all_simulations.(simulated_model_feature_set).(sprintf('sub_%i',sim_index));
% params = all_params.(simulated_model_feature_set).(sprintf('sub_%i',sim_index));

    global Ss Ts
    
    % Get data
    pred_grid = nan(4,4,4);
    for block = 0:3
        for S = Ss
            for T = Ts
                choice = sim_dat{sim_dat{:,'Block'}==block & sim_dat{:,'S'}==S & sim_dat{:,'T'}==T,'GivenAns'};
                conf = sim_dat{sim_dat{:,'Block'}==block & sim_dat{:,'S'}==S & sim_dat{:,'T'}==T,'Confidence'};
                if strcmp(choice,'coop')
                    choice_conf = 0.5 + 0.5 * conf;
                else
                    choice_conf = 0.5 - 0.5 * conf;
                end
                pred_grid(block+1,Ts==T,Ss==S) = choice_conf;
            end
        end
    end

    % Plot
    hex = flipud(['#B2182B'; '#D6604D'; '#F4A582'; '#FDDBC7'; '#FFFFFF'; '#D1E5F0'; '#92C5DE'; '#4393C3'; '#2166AC']);
    vec = (100:-(100/(length(hex)-1)):0)';
    raw = sscanf(hex','#%2x%2x%2x',[3,size(hex,1)]).' / 255;
    N = 128;
    cmap = interp1(vec,raw,linspace(100,0,N),'pchip');

    figure(1);
    colormap(cmap);
    for i = 1:4
        subplot(1,4,i)
        imagesc(reshape(pred_grid(i,:,:),4,4),[0,1]);
        axis equal;
        axis tight;
        title(unique(sim_dat{sim_dat{:,'Block'}==(i-1),'Type_Total'}));
    end

end

