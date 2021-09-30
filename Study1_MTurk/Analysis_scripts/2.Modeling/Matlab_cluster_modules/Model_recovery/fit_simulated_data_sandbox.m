smi = 3;
sim_index = 1;
%% Prep
addpath('..');
addpath('../helpers');
define_globals;

% Define model function
model_function = @cost_function_featureLearner_6;
model_feature_sets = {'CoHgShSgPd','CoGrRiNa'};

%% Select simulation data
load('simulations.mat');
load('params.mat');
simulated_model_feature_set = model_feature_sets{smi};
sim_dat = all_simulations.(simulated_model_feature_set).(sprintf('sub_%i',sim_index));
params = all_params.(simulated_model_feature_set).(sprintf('sub_%i',sim_index));

%% Open results file
fid = fopen(sprintf('results/results_smi-%i_sim_index-%i.txt',smi,sim_index),'w');
header = sprintf('smi,sim_index,mfsi,model_feature_set,params_original,params_recovered,SSE,BIC\n');
fwrite(fid,header);

%% Fit models
options = optimoptions('fmincon');
options.Display = 'iter';

for mfsi = 1:length(model_feature_sets)

    model_feature_set = model_feature_sets{mfsi};
    disp(model_feature_set);

    % Find model details
    [basis_set,~] = define_basis_set_2(model_feature_set);
    n_features_full_basis_set = length(basis_set);

    % Parameter bounds
    param_bounds = [repmat([0.01   , 10],(1 + double(0)),1); ... % LR
                   0.01   , 5; ... % InvTemp
                   repmat([0 - (1 - double(0)) * 20,20], n_features_full_basis_set,1); ... % Weights
                   ];
    lb = param_bounds(:,1);
    ub = param_bounds(:,2);

    % Fit
    objective_function = @(params)model_function(params, ... % Free parameters
        [], sim_dat, [], ... % Input data
        basis_set, 0, 0, 'joint', ... % Model structure
        0, 0, false);
    params0 = lb + rand(size(param_bounds,1),1).*(ub-lb);
    [x,cost] = fmincon(objective_function, params0, ...
        [], [], [], [], lb, ub, [], options);
    % Get BIC
    model_out = model_function(x, ... % Free parameters
        [], sim_dat, [], ... % Input data
        basis_set, 0, 0, 'joint', ... % Model structure
        0, 0, true);
    SSE = model_out.SSE;
    if cost ~= SSE
        error('Something went wrong: SSE not consistent across model runs');
    end
    LL = -model_out.NLL;
    BIC = model_out.BIC;

    % Store
    writeline = sprintf('%i,%i,%i,%s,%.2f,%.2f\n',...
        smi,sim_index,mfsi,model_feature_set,params,x,SSE,BIC);
    fwrite(fid,writeline);

end