%% POSTERIOR PREDICTIVE CHECKS FOR MODEL
%
% -------------------------------------------------------------------------
% Goals:
% -------------------------------------------------------------------------
%
% 1. Plot & store posterior predictive checks per subject, including:
%   * Predictions
%   * Confidence

% Revised goals:
% * Show for a few subjects how a subset model (their best subset) is better than full
% model. This should justify our approach.
% * Show for a few subjects how their best subset is better than another
% subset. Examples
% - Subject 32. Has Nash motive. Removing this or replacing by Risk leads
% to worse fit and lack of clear Nash component in Pess block.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load functions
addpath('helpers');
% Game variables; softmax choice rule
define_globals;
% Load data
[game_dat, sub_IDs] = load_game_dat;
% Load model results
study_name = 'Study1_Mturk';
date_string = '2020-03-24';
features = 'CoGrRiNa';
model_string = 'FeatureRL';
fit_to = 'joint';
niter = 10;
gaze = 'False';
filename = ['/Users/jeroen/Dropbox (Brown)/Postdoc FHL/JEROEN/SOC_STRUCT_LEARN/',...
         'Study1_MTurk/Data/Cleaned/Model_results/',...
         sprintf('%s_%s_%s_fitto-%s_gaze-%s_niter-%i.csv',...
             model_string,features,date_string, fit_to, gaze, niter)];
opts = detectImportOptions(filename);
opts = setvartype(opts,{'sub_ind'},'single');
opts = setvartype(opts,{'sub_ID','comb','fit_to'},'string');
clear model_fits
model_results = readtable(filename,opts,'ReadRowNames',false);

% Preallocate
out_all = {};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INDIVIDUAL PCC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

sub_index = 32;
sub_ID = sub_IDs(sub_index);
sub_dat = game_dat(game_dat.subID==sub_ID,:);
sub_model = model_results(model_results.sub_ID == sub_ID,:);
best_model = sub_model(sub_model.BIC == min(sub_model.BIC),:);
comb = best_model.comb;

% Get features
features_string = best_model.feature_names{1};
features_split = strsplit(features_string,'_');
n_features = length(features_split);
features = '';
for fi = 1:n_features
    features = [features, features_split{fi}(1:2)];
end

% Build basis set for model
[basis_sub_set,~] = define_basis_set(features);
basis_sub_set_names = strip(sprintf('%s_',basis_sub_set.name),'_');
fprintf('Basis set: %s.\n',basis_sub_set_names);

% Get params
params = table2array(best_model(:,[10,12,14,15,16,17]))';

% Run model
visualize = false;
gaze = false; return_info = true;
out = cost_function_featureLearner_6(params, ... % Free parameters
    sub_ID, sub_dat, [], ... % Input data
    basis_sub_set, 0, 0, fit_to, ... % Model structure
    gaze, visualize, return_info);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% INDIVIDUAL PCC & SHOW MODEL COMPUTATIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Settings
sub_index = 32;
comb = '111000';
asymmetric_LR = [];
bounded_weights = [];

% Game data
sub_ID = sub_IDs(sub_index);
fprintf('Sub index: %i, sub ID: %s\n',sub_index, sub_ID);
sub_dat = game_dat(game_dat.subID==sub_ID,:);

% Model data
model_use = select_model(model_results, sub_index, comb, asymmetric_LR, bounded_weights);

% % Edit model
% model_use.inv_temp = 25;

% Run model
[out, features] = run_model(sub_ID, sub_dat, model_use, []);

% Visualize
plot_PCC_subject(sub_index, sub_ID, sub_dat, out, features)
out.sub_index = sub_index;
out.sub_ID = sub_ID;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTIONS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% High level function to run PCC for a subject

function out_all = run_PCC(cfg, sub_index, game_dat, model_results, out_all)

    % Read config inputs
    sub_IDs = cfg.sub_IDs;
    comb = cfg.comb;
    asymmetric_LR = cfg.asymmetric_LR;
    bounded_weights = cfg.bounded_weights;

    % Game data
    sub_ID = sub_IDs(sub_index);
    fprintf('Sub index: %i, sub ID: %s\n',sub_index, sub_ID);
    sub_dat = game_dat(game_dat.subID==sub_ID,:);

    % Model data
    model_use = select_model(model_results, sub_index, comb, asymmetric_LR, bounded_weights);

    % Run model
    [out, features] = run_model(sub_ID, sub_dat, model_use, []);

    % Visualize

    plot_PCC_subject(sub_index, sub_ID, sub_dat, out, features)
    out.sub_index = sub_index;
    out.sub_ID = sub_ID;
    fprintf('\n\n');
    
    % Store
    if ~isempty(out_all)
        out_all{sub_index,1} = out;
    else
        out_all = out;
    end

end

%% Load model results function

function model_results = load_model_results(study_name, model_date_string)

    base_dir = '/Users/jvanbaar/Dropbox (Brown)/Postdoc FHL/JEROEN/SOC_STRUCT_LEARN/';
    model_results_fname = [base_dir, sprintf('%s/Data/Cleaned/',study_name), ...
        sprintf('model_results_%s.csv',model_date_string)];
    opts = detectImportOptions(model_results_fname);
    opts = setvartype(opts,{'sub_ind','comb_index','asymm_LR','bounded_weights'},'single');
    opts = setvartype(opts,{'sub_ID','comb'},'string');
    model_results = readtable(model_results_fname, opts);

end

%% Select model function

function model_use = select_model(model_results, sub_index, comb, asymmetric_LR, bounded_weights)

    % Model data
    subject_model_rows = model_results.sub_ind == sub_index;
    subject_model_results = model_results(subject_model_rows,:);

    %% Select models
    if ~isempty(comb)
        subject_model_results = subject_model_results(...
              strcmp(subject_model_results.comb,comb),:);
          fprintf('Using custom comb: %s.\n',comb);
    end
    if ~isempty(asymmetric_LR)
        subject_model_results = subject_model_results(...
              subject_model_results.asymm_LR == asymmetric_LR,:);
          fprintf('Using custom asymmetric_LR: %i.\n',double(asymmetric_LR));
    end
    if ~isempty(bounded_weights)
        subject_model_results = subject_model_results(...
              subject_model_results.bounded_weights == bounded_weights,:);
          fprintf('Using custom bounded_weights: %i.\n',double(bounded_weights));
    end

    %% Find best model within possibly constrained set

    subject_model_results = sortrows(subject_model_results,{'BIC'});
    model_use = subject_model_results(1,:);

end

%% Run model function

function [out, features] = run_model(sub_ID, sub_dat, model_use, visualize)

    % Get model structure
    fit_to = model_use.fit_to;
    comb = model_use.comb;
    asymmetric_LR = model_use.asymm_LR;
    bounded_weights = model_use.bounded_weights;
    fprintf('Using model structure: comb %s, asymm_LR %i, bounded_weights %i.\n',...
            comb, double(asymmetric_LR), double(bounded_weights));

    % Get features
    features_string = model_use.feature_names{1};
    features_split = strsplit(features_string,'_');
    n_features = length(features_split);
    features = '';
    for fi = 1:n_features
        features = [features, features_split{fi}(1:2)];
    end

    % Build basis set for model
    [basis_sub_set,~] = define_basis_set(features);
    basis_sub_set_names = strip(sprintf('%s_',basis_sub_set.name),'_');
    fprintf('Basis set: %s.\n',basis_sub_set_names);

    % Get params
    params = table2array(model_use(:,9:17))';
    fprintf('Model parameters: '); fprintf('%.2f, ',params); fprintf('\n');
    if asymmetric_LR == 0
        params = params([1,3:end]);
    end
    
    % Run model
    if isempty(visualize)
        visualize = false;
    end
    gaze = false; return_info = true;
    out = cost_function_featureLearner_4(params, ... % Free parameters
        sub_ID, sub_dat, ... % Input data
        basis_sub_set, bounded_weights, asymmetric_LR, fit_to, ... % Model structure
        gaze, visualize, return_info);
    fprintf('SSE = %.2f.\n',out.SSE);

end

%% Visualize function
function plot_PCC_subject(sub_index, sub_ID, sub_dat, out, features)
    
    define_globals;

    figure(sub_index);
    clf(sub_index);
    nrows = 4; ncols = 4;

    for block = 1:4

        % Data
        block_dat = sub_dat(sub_dat.Block==(block-1),...
            {'Trial','S','T','CorrAns','GivenAns','Confidence','Type_Total'});
        player_type = char(unique(block_dat.Type_Total));
        player_type =[player_type(1:find(player_type=='_')-1) '\_' player_type(find(player_type=='_')+1:end)];

        % True & predicted
        player_choices = zeros(4,4);
        for ti = 1:16
            S = block_dat.S(ti); T = block_dat.T(ti);
            player_choices(Ss == S, Ts == T) = strcmp(block_dat.CorrAns{ti},'coop'); % Write as -1, 1
        end

        subplot(nrows,ncols,block);
        imagesc('CData',player_choices);
        caxis([0,1]);
        xlabel('T'); ylabel('S');
        xticks(1:11); yticks(1:11);
        xticklabels(Ts); yticklabels(Ss);
        title(sprintf('Block %i: %s',block,player_type));
        axis equal;
        axis tight;
        colorbar;

        subplot(nrows,ncols,4 + block);
        imagesc('CData',out.participant_prediction_confidence_joint_grid_per_block.(sprintf('block_%i',block)));
        caxis([0,1]);
        xlabel('T'); ylabel('S');
        xticks(1:11); yticks(1:11);
        xticklabels(Ts); yticklabels(Ss);
        title(sprintf('sub %s prediction',sub_ID));
        axis equal;
        axis tight;
        colorbar;

        subplot(nrows,ncols,8 + block);
        imagesc('CData',out.model_coop_likelihoods_per_block.(sprintf('block_%i',block)));
        caxis([0,1]);
        xlabel('T'); ylabel('S');
        xticks(1:11); yticks(1:11);
        xticklabels(Ts); yticklabels(Ss);
        title(sprintf('Model %s',features));
        axis equal;
        axis tight;
        colorbar;

        subplot(nrows,ncols,12 + block);
        % Learning curve - model
        n_perms = 100;
        lc_model = nan(16,n_perms);
        for i = 1:n_perms
            preds = rand(16,1) <= out.store_pred_dist(1 + (block-1) * 16 : 16 + (block-1) * 16,1);
            lc_model(:,i) = cumsum(strcmp(block_dat.CorrAns,'coop') == preds);
        end
        model_lc_range = [mean(lc_model') - std(lc_model'), fliplr(mean(lc_model') + std(lc_model'))];
        % Learning curve - true
        learning_curve = cumsum(block_dat.CorrAns == block_dat.GivenAns);
        % Plot
        plot(1:16, learning_curve, 'b', 'LineWidth', 2);
        patch([1:16,16:-1:1], model_lc_range, 'g', 'FaceAlpha', 0.4);
        xlim([1,16]); ylim([0,16]);
        ylabel('Cumulative score'); xlabel('Trial');
        mean_trial_prob = mean(out.choice_probabilities_model(1 + (block-1) * 16 : 16 + (block-1) * 16,1));
        title(sprintf('Mean response prob. = %.2f', mean_trial_prob)); %learning_curve(end)/16*100));

    end
end