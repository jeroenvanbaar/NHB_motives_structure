%%
% Goal: simulate data from model

function feature_learner_simulate(sub_index, n_sims, model_string, features, pred_type)
    
    % -------------------------------------------------------------------------
    % Inputs to constrain model types considered:
    % -------------------------------------------------------------------------
    %
    % ----- REQUIRED inputs -----
    % 'pred_type' = 'point' or 'sampledistr'
    %
    % ----- OPTIONAL inputs -----
    % none
    % 
    
    %% Load functions
    addpath('helpers');

    %% Game variables; softmax choice rule
    define_globals;
    
    %% Load model fits
    if sub_index < 151
        study_name = 'Study1_Mturk';
        date_string = '2020-03-24';
    elseif (sub_index > 150) && (sub_index < 201)
        study_name = 'Study2_EyeTracking';
        date_string = '2020-03-31';
    elseif (sub_index > 200)
        study_name = 'Study4_NewPlayerTypes';
        date_string = '2021-04-09';
    end
    fit_to = 'joint';
    niter = 10;
    gaze = 'false';
    filename = ['/gpfs_home/jvanbaar/data/jvanbaar/SOC_STRUCT_LEARN/',...
        'ComputationalModel/FeatureRL/Results/final_model_results_for_paper/',...
        study_name, sprintf('/%s_%s_%s_fitto-%s_gaze-%s_niter-%i.csv',...
        model_string,features,date_string, fit_to, gaze, niter)];
    opts = detectImportOptions(filename);
    opts = setvartype(opts,{'sub_ind'},'single');
    opts = setvartype(opts,{'sub_ID','comb','fit_to'},'string');
    clear model_results
    model_results = readtable(filename,opts,'ReadRowNames',false);
    
    %% Load game data
    [game_dat, sub_IDs] = load_game_dat;
    sub_ID = sub_IDs(sub_index);
    fprintf('Game data loaded with sub ID: %s\n',sub_ID);
    sub_dat = game_dat(game_dat.subID==sub_ID,:);
    
    %% Specify model for subject
    sub_model = model_results(model_results.sub_ID == sub_ID,:);
    best_model = sub_model(sub_model.BIC == min(sub_model.BIC),:);
    comb = best_model.comb;

    % Get features
    features_string = best_model.feature_names{1};
    features_split = strsplit(features_string,'_');
    n_features = length(features_split);
    sub_features = '';
    for fi = 1:n_features
        sub_features = [sub_features, features_split{fi}(1:2)];
    end

    % Build basis set for model
    [basis_sub_set,~] = define_basis_set_2(sub_features);
    basis_sub_set_names = strip(sprintf('%s_',basis_sub_set.name),'_');
    
    % Get params
%     params = table2array(best_model(:,[10,12,14,15,16,17]))';
    params = table2array(best_model(:,{'LR_up','inv_temp', 'feature_weight_1',...
        'feature_weight_2', 'feature_weight_3', 'feature_weight_4'}))';
    
    %% Open log / results files
    tmp = what('../../../..');
    data_dir = [tmp.path, '/data/jvanbaar/SOC_STRUCT_LEARN/ComputationalModel/FeatureRL/Simulations'];
    results_dir = sprintf('%s/%s_%s_%s_%s',data_dir,study_name,model_string,features,date_string);
    fprintf('Using results directory %s\n',results_dir);
    if ~exist(results_dir,'dir')
        mkdir(results_dir);
        fprintf('Created results directory %s\n',results_dir);
    end
    log_filename_base = sprintf('%s/simulations_subInd-%i_subID-%s_pred-%s_n-sims-%i.csv',...
        results_dir, sub_index, sub_ID, pred_type, n_sims);
    filename = log_filename_base;
    suffix = 1;
    while exist(filename,'file') == 2
        suffix = suffix + 1;
        filename = [log_filename_base(1:end-4) sprintf('_%i.csv',suffix)];
    end
    log_ID = fopen(filename,'a');
    header_line = 'sub_ind,sub_ID,sim_index,block,trial,S,T,player_type,player_choice,sub_pred,model_pred\n';
    fprintf(log_ID, header_line);
    
    %% Run simulations
    visualize = false;
    gaze = false; return_info = true;
    for i = 1:n_sims
        fprintf('%i,',i);
        out = cost_function_featureLearner_6(params, ... % Free parameters
            sub_ID, sub_dat, [], ... % Input data
            basis_sub_set, 0, 0, 'joint', ... % Model structure
            gaze, visualize, return_info);
%         model_pred = out.store_model_prediction;
        blocks = sub_dat.Block;
        player_types = sub_dat.Type_Total;
        ss = sub_dat.S;
        ts = sub_dat.T;
        player_choices = sub_dat.CorrAns=="coop";
        if strcmp(pred_type, 'point')
            model_pred = out.store_pred_dist(:,1) > .5;
        elseif strcmp(pred_type, 'sampledistr')
            model_pred = out.store_sampled_model_prediction;
        end
        sub_pred = out.store_participant_prediction_confidence_joint > .5;
        n_obs = length(model_pred);
        for ti = 1:n_obs
            bt = mod(ti-1,16)+1;
            fprintf(log_ID, '%i,%s,%i,%i,%i,%i,%i,%s,%i,%i,%i\n', sub_index, sub_ID, i, ...
                blocks(ti), bt, ss(ti), ts(ti), player_types(ti), ...
                player_choices(ti), sub_pred(ti), model_pred(ti));
        end
    end
fprintf('\nDone.');
end
