%% Compare model without and with gaze

% Load functions
addpath('helpers');
% Game variables; softmax choice rule
define_globals;
% Define basis set
[basis_set,combs] = define_basis_set(features);
n_features_full_basis_set = length(basis_set);
comb_index_list = 1:length(combs);
if comb_index_choice > 0 % This means one comb index was input by user
    comb_index_list = comb_index_choice;
end

%% Select subject
sub_index = 158;

% Load data
[game_dat, sub_IDs] = load_game_dat;
sub_ID = sub_IDs(sub_index);
fprintf('Game data loaded with sub ID: %s\n',sub_ID);
sub_dat = game_dat(game_dat.subID==sub_ID,:);

% Load gaze data
if gaze
    [gaze_dat, ~] = load_gaze_dat;
    sub_gaze_dat = gaze_dat(gaze_dat.sub == str2double(sub_ID) - 5000,:);
    trials = unique(sub_gaze_dat.trial)';
    trial_gaze_diffs = nan(length(trials),1);
    for trial = trials
        trial_gaze_dat = sub_gaze_dat(sub_gaze_dat.trial == trial,...
            {'trial','num_S_T','dur_pct'});
        S_actual = sum(trial_gaze_dat{strcmp(trial_gaze_dat.num_S_T, "S"),'dur_pct'});
        T_actual = sum(trial_gaze_dat{strcmp(trial_gaze_dat.num_S_T, "T"),'dur_pct'});
        if S_actual == 0 & T_actual == 0
            gaze_diff = 0;
        else
            gaze_diff = (S_actual - T_actual) / (S_actual + T_actual);
        end
        trial_gaze_diffs(trial) = gaze_diff;
    end
    if length(trials) == 0
        error(sprintf('No gaze data present for subject %s.\n',sub_ID));
    else
        fprintf('Gaze data loaded for %i trials.\n', length(trials));
    end
    trial_gaze_diffs(isnan(trial_gaze_diffs)) = 0;
else
    trial_gaze_diffs = [];
end

%% Load model results
tmp = what('../../../..');
data_dir = [tmp.path, '/data/jvanbaar/SOC_STRUCT_LEARN/ComputationalModel/FeatureRL'];
results_dir = sprintf('%s/Results/results_%s',data_dir,'2019-12-04');
n_iter = 10;

% Gaze-true:
log_filename_base = sprintf('%s/subInd-%i_subID-%s_fitto-%s_features-%s_gaze-%s_niter-%i_log.csv',...
        results_dir, sub_index, sub_ID, fit_to, features, 'true', n_iter);
opts = detectImportOptions(log_filename_base);
opts = setvartype(opts,{'sub_ind','comb_index','asymm_LR','bounded_weights','gaze','LR_up','LR_down',...
    'inv_temp','gaze_bias','feature_weight_1','feature_weight_2','feature_weight_3','feature_weight_4',...
    'SSE','LL','BIC'},'single');
opts = setvartype(opts,{'sub_ID','fit_to', 'comb','feature_names',...
    'cost_type'},'string');
clear model_dat
model_dat_g = readtable(log_filename_base,opts,'ReadRowNames',false);

% Gaze-false:
log_filename_base = sprintf('%s/subInd-%i_subID-%s_fitto-%s_features-%s_gaze-%s_niter-%i_log.csv',...
        results_dir, sub_index, sub_ID, fit_to, features, 'false', n_iter);
opts = detectImportOptions(log_filename_base);
opts = setvartype(opts,{'sub_ind','comb_index','asymm_LR','bounded_weights','gaze','LR_up','LR_down',...
    'inv_temp','gaze_bias','feature_weight_1','feature_weight_2','feature_weight_3','feature_weight_4',...
    'SSE','LL','BIC'},'single');
opts = setvartype(opts,{'sub_ID','fit_to', 'comb','feature_names',...
    'cost_type'},'string');
clear model_dat
model_dat_ng = readtable(log_filename_base,opts,'ReadRowNames',false);

% Total
model_dat = [model_dat_g; model_dat_ng];

[best_BIC_g,ind] = min(model_dat_g.BIC);
best_model_g = model_dat_g(ind,:);
[best_BIC_ng,ind] = min(model_dat_ng.BIC);
best_model_ng = model_dat_ng(ind,:);
fprintf('No gaze: best BIC = %.2f. Gaze: best BIC = %.2f.\n',best_BIC_ng, best_BIC_g);

%% Function
PPC_fun = @(basis_sub_set, bounded_weights, gaze, params) cost_function_featureLearner_5(params, ... % Free parameters
            sub_ID, sub_dat, trial_gaze_diffs, ... % Input data
            basis_sub_set, bounded_weights, asymmetric_LR, fit_to, ... % Model structure
            gaze, false, true);
        
% Run
comb_ng = char(best_model_ng{1,'comb'});
basis_sub_set_ng = basis_set(comb_ng=='1');
bw_ng = best_model_ng{1,'bounded_weights'};
params_ng = best_model_ng{1,{'LR_up','inv_temp',...
    'feature_weight_1','feature_weight_2','feature_weight_3','feature_weight_4'}}';
comb_g = char(best_model_g{1,'comb'});
basis_sub_set_g = basis_set(comb_g=='1');
bw_g = best_model_g{1,'bounded_weights'};
params_g = best_model_g{1,{'LR_up','inv_temp','gaze_bias',...
    'feature_weight_1','feature_weight_2','feature_weight_3','feature_weight_4'}}';
out_ng = PPC_fun(basis_sub_set_ng, bw_ng, false, params_ng);
out_g = PPC_fun(basis_sub_set_g, bw_g, true, params_g);

%% Plot
fig = figure;
nrows = 6;
ncols = 4;

player_types = {'opt_nat','pess_nat','opt_inv','pess_inv'};
for ri = 1:2
    for pti = 1:4
        
        player_type = player_types{pti};
        block_nrs = unique(sub_dat{strcmp(sub_dat.Type_Total,player_type),'Block'})';
        block = block_nrs(ri)+1;
        
        sub_predictions = ...
            out_ng.participant_prediction_confidence_joint_grid_per_block.(sprintf('block_%i',block));
        model_predictions_ng = ...
            out_ng.model_coop_likelihoods_per_block.(sprintf('block_%i',block));
        model_predictions_g = ...
            out_g.model_coop_likelihoods_per_block.(sprintf('block_%i',block));
        
        % Sub predictions (top row)
        subplot(nrows,ncols,pti + (ri-1)*3*ncols);
        imagesc('CData',sub_predictions);
        caxis([0,1]);
        xlabel('T'); ylabel('S');
        xticks(1:11); yticks(1:11);
        xticklabels(Ts); yticklabels(Ss);
        title(sprintf('%i: %s - s%i',block,replace(player_type,'_','\_'), sub_index));
        axis equal;
        axis tight;
        colorbar;
        
        % Model without gaze (middle row)
        subplot(nrows,ncols,pti + ncols + (ri-1)*3*ncols);
        imagesc('CData',model_predictions_ng);
        caxis([0,1]);
        xlabel('T'); ylabel('S');
        xticks(1:11); yticks(1:11);
        xticklabels(Ts); yticklabels(Ss);
        title(sprintf('%i: %s - ng',block,replace(player_type,'_','\_')));
        axis equal;
        axis tight;
        colorbar;
        
        
        % Model with gaze (middle row)
        subplot(nrows,ncols,pti + 2 * ncols + (ri-1)*3*ncols);
        imagesc('CData',model_predictions_g);
        caxis([0,1]);
        xlabel('T'); ylabel('S');
        xticks(1:11); yticks(1:11);
        xticklabels(Ts); yticklabels(Ss);
        title(sprintf('%i: %s - g',block,replace(player_type,'_','\_')));
        axis equal;
        axis tight;
        colorbar;
        
    end
end
        
%% Run all-subject model comparison

tmp = what('../../../..');
data_dir = [tmp.path, '/data/jvanbaar/SOC_STRUCT_LEARN/ComputationalModel/FeatureRL'];
results_dir = sprintf('%s/Results/results_%s',data_dir,'2019-12-04');
n_iter = 10;

model_results = [];
for sub_index = 151:200
    fprintf('%i, ',sub_index);
    sub_ID = sub_IDs(sub_index);
    try
        
        % Gaze-true:
        log_filename_base = sprintf('%s/subInd-%i_subID-%s_fitto-%s_features-%s_gaze-%s_niter-%i_log.csv',...
                results_dir, sub_index, sub_ID, fit_to, features, 'true', n_iter);
        opts = detectImportOptions(log_filename_base);
        opts = setvartype(opts,{'sub_ind','comb_index','asymm_LR','bounded_weights','gaze','LR_up','LR_down',...
            'inv_temp','gaze_bias','feature_weight_1','feature_weight_2','feature_weight_3','feature_weight_4',...
            'SSE','LL','BIC'},'single');
        opts = setvartype(opts,{'sub_ID','fit_to', 'comb','feature_names',...
            'cost_type'},'string');
        clear model_dat
        model_dat_g = readtable(log_filename_base,opts,'ReadRowNames',false);

        % Gaze-false:
        log_filename_base = sprintf('%s/subInd-%i_subID-%s_fitto-%s_features-%s_gaze-%s_niter-%i_log.csv',...
                results_dir, sub_index, sub_ID, fit_to, features, 'false', n_iter);
        opts = detectImportOptions(log_filename_base);
        opts = setvartype(opts,{'sub_ind','comb_index','asymm_LR','bounded_weights','gaze','LR_up','LR_down',...
            'inv_temp','gaze_bias','feature_weight_1','feature_weight_2','feature_weight_3','feature_weight_4',...
            'SSE','LL','BIC'},'single');
        opts = setvartype(opts,{'sub_ID','fit_to', 'comb','feature_names',...
            'cost_type'},'string');
        clear model_dat
        model_dat_ng = readtable(log_filename_base,opts,'ReadRowNames',false);

        % Total
        model_dat = [model_dat_g; model_dat_ng];

        best_SSE_g = min(model_dat_g.SSE);
        best_BIC_g = min(model_dat_g.BIC);
        best_SSE_ng = min(model_dat_ng.SSE);
        best_BIC_ng = min(model_dat_ng.BIC);
        best_comb_g = model_dat_g{model_dat_g.BIC==best_BIC_g,'comb_index'}(1);
        best_comb_ng = model_dat_ng{model_dat_ng.BIC==best_BIC_ng,'comb_index'}(1);
        model_results = [model_results; [sub_index, best_comb_ng, best_comb_g, ...
            best_SSE_ng, best_SSE_g, best_BIC_ng, best_BIC_g]];
        
    catch
        fprintf('Skipped subject %i\n',sub_index);
    end
end
fprintf('\n');

%% Plot
figure;
% subplot(1,3,1);
plot(model_results)
subplot(1,2,1);
SSE_change = model_results(:,5) - model_results(:,4);
histogram(SSE_change, 20);
title('SSE'); ylabel('Gaze model ? no-gaze model');
subplot(1,2,2);
histogram(model_results(:,7) - model_results(:,6), 20);
title('BIC (lower = better)'); ylabel('Gaze model ? no-gaze model');

most_improved_subject = 150 + find(SSE_change == min(SSE_change));

figure;
histogram(model_results(:,3),0.5:1:15.5)




