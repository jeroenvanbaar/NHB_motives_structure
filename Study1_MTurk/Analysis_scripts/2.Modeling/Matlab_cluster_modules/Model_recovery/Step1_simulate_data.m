%%
% Script to run model recovery / confusion on the three feature RL models
% and the two Bayesian category learning models (motives, players) for the
% social structure learning paper

% Plan:
% 1. Simulate data for the first 100 subjects using each of the three models.
% 2. Fit each model to each dataset.
% 3. Count how often true model is best.

%% Prep
addpath('../FeatureRL/helpers');
define_globals;
[game_dat, sub_IDs] = load_game_dat;

% How many pseudo-subjects to generate?
n_subs = 100;
get_params_from = 'subjects'; % Can be 'subjects' or 'uniform'

% Preallocate
all_simulations = struct;
all_params = struct;
plot_model_predictions = false;

%% 1. Simulate data - feature RL
model_type = 'FeatureRL';
model_feature_sets = {'CoST','CoHgShSgPd','CoGrRiNa'};

model_date_dict = struct();
model_date_dict.CoST = '2020-03-26';
model_date_dict.CoHgShSgPd = '2020-03-24';
model_date_dict.CoGrRiNa = '2020-03-24';

addpath(sprintf('../%s/',model_type));
model_function = @cost_function_featureLearner_6;

for mfsi = 1:length(model_feature_sets)
    model_feature_set = model_feature_sets{mfsi};
    disp(model_feature_set);
    
    all_simulations.(model_feature_set) = struct;
    all_params.(model_feature_set) = struct;
    
    % Find model details
    [basis_set,~] = define_basis_set_2(model_feature_set);
    n_features_full_basis_set = length(basis_set);
    asymmetric_LR = 0;
    bounded_weights = 0;
    gaze = 0;
    
    % Parameter bounds
    param_bounds = [repmat([0.01   , 10],(1 + double(asymmetric_LR)),1); ... % LR
                   0.01   , 5; ... % InvTemp
                   repmat([0,10],double(gaze),1); ... % Gaze bias
                   repmat([0 - (1 - double(bounded_weights)) * 20,20],...
                        length(basis_set),1); ... % Weights
                   ];
    lb = param_bounds(:,1);
    ub = param_bounds(:,2);
    
    % Select sub
    for sub_index = 1:n_subs
        sub_ID = sub_IDs(sub_index);
        sub_dat = game_dat(game_dat.subID==sub_ID,:);

        % Params
        switch get_params_from
            case 'uniform'
                params = lb + rand(size(param_bounds,1),1).*(ub-lb);
            case 'subjects'
                loaded_params = load_fitted_params(model_type,model_date_dict.(model_feature_set),sub_ID,model_feature_set);
                params = [loaded_params.LR; loaded_params.inv_temp;...
                    loaded_params.feature_weights'];
        end
        
        % Run
        model_out = model_function(params, ... % Free parameters
                        sub_ID, sub_dat, [], ... % Input data
                        basis_set, 0, 0, 'joint', ... % Model structure
                        0, 0, true);
        if plot_model_predictions == true
            figure;
            for i = 1:4
                subplot(1,4,i)
                imagesc(model_out.model_coop_likelihoods_per_block.(sprintf('block_%i',i)),[0,1]);
            end
        end

        % Tabulate simulation data
        sim_dat = sub_dat;
        for trial = 1:64
            coop_pred = model_out.store_pred_dist(trial,1);
            givenans = coop_pred > 0.5; % 0 defect, 1 coop
            if givenans == 0
                sim_dat.GivenAns(trial) = "def";
                sim_dat.Confidence(trial) = (0.5 - coop_pred) * 2;
            else
                sim_dat.GivenAns(trial) = "coop";
                sim_dat.Confidence(trial) = (coop_pred - 0.5) * 2;
            end
        end
        all_simulations.(model_feature_set).(sprintf('sub_%i',sub_index)) = sim_dat;
        all_params.(model_feature_set).(sprintf('sub_%i',sub_index)) = params;
    end
end

%% Simulate -- Bayesian models
model_type = 'BayesianIO';
model_feature_sets = {'motives','players'};

model_date_dict = struct();
model_date_dict.motives = '2021-05-15';
model_date_dict.players = '2021-05-15';

addpath(genpath(sprintf('../%s/',model_type)));
model_function = @NLL_SocPredIdealObs;

for mfsi = 1:length(model_feature_sets)
    model_feature_set = model_feature_sets{mfsi};
    disp(model_feature_set);
    
    all_simulations.(model_feature_set) = struct;
    all_params.(model_feature_set) = struct;
    
    % Find model details
    switch model_feature_set
        case 'players'
            stateNames  = {'Opt','Pess','InvOpt','InvPess'};
        case 'motives'
            stateNames  = {'Trust','Opt','Pess','Nash'};
        case 'poncela'
            stateNames  = {'Trust','Opt','Pess','Env'};
    end
    
    % Parameter bounds
    paramBounds = repmat([0,1],length(stateNames),1);
    lb = paramBounds(:,1);
    ub = paramBounds(:,2);
    
    % Select sub
    for sub_index = 1:n_subs
        sub_ID = sub_IDs(sub_index);
        sub_dat = game_dat(game_dat.subID==sub_ID,:);

        % Params
        switch get_params_from
            case 'uniform'
                params = nan(size(paramBounds,1),1);
                initPriors = rand(4,1);
                initPriors = initPriors ./ sum(initPriors);
                params(1:3) = initPriors(1:3); % This ensures that the sum of the priors does not exceed 1, allowing the fourth prior to be computed
                params(4) = lb(4) + rand(1,1).*ub(4);
            case 'subjects'
                loaded_params = load_fitted_params(model_type,model_date_dict.(model_feature_set),sub_ID,model_feature_set);
                params = [loaded_params.priors';loaded_params.noise];
        end

        % Run
        model_out = model_function(params, sub_ID, sub_dat, ...
            stateNames, 'joint', 1, 0);

        % Tabulate simulation data
        sim_dat = sub_dat;
        for trial = 1:64
            coop_pred = model_out.storePredDist(trial,2);
            givenans = coop_pred > 0.5; % 0 defect, 1 coop
            if givenans == 0
                sim_dat.GivenAns(trial) = "def";
                sim_dat.Confidence(trial) = (0.5 - coop_pred) * 2;
            else
                sim_dat.GivenAns(trial) = "coop";
                sim_dat.Confidence(trial) = (coop_pred - 0.5) * 2;
            end
        end
        all_simulations.(model_feature_set).(sprintf('sub_%i',sub_index)) = sim_dat;
        all_params.(model_feature_set).(sprintf('sub_%i',sub_index)) = params;
    end
end


%% Store
save('simulations.mat','all_simulations');
save('params.mat','all_params');
