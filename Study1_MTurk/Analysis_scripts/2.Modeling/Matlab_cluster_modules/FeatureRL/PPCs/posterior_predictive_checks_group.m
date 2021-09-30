%% POSTERIOR PREDICTIVE CHECKS FOR MODEL
%
% -------------------------------------------------------------------------
% Goal: Predict predictions+confidence trial-by-trial using fitted models
% -------------------------------------------------------------------------
%
% Steps:
% 1. Select model (either full or best per subject)
% 2. Run model again
% 3. Record predicted behavior
% 4. Store
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load functions
addpath(genpath('helpers'));
% Game variables; softmax choice rule
define_globals;
% Load data
[game_dat, sub_IDs] = load_game_dat;

%% Load model results
% model_results_file = ['/Users/jeroen/Dropbox (Brown)/Postdoc FHL/JEROEN/SOC_STRUCT_LEARN/',...
%     'Study1_Mturk/Data/Cleaned/Model_results/FeatureRL_CoGrRiNa_2020-03-24_fitto-joint_gaze-false_niter-10.csv'];
% model_results_file = ['/Users/jeroen/Dropbox (Brown)/Postdoc FHL/JEROEN/SOC_STRUCT_LEARN/',...
%     'Study2_EyeTracking/Data/Cleaned/Model_results/FeatureRL_CoGrRiNa_2020-03-31_fitto-joint_gaze-false_niter-10.csv'];
model_results_file = ['/gpfs/home/jvanbaar/data/jvanbaar/SOC_STRUCT_LEARN/ComputationalModel/FeatureRL',...
    '/Results/results_2021-04-09/FeatureRL_CoGrRiNa_2021-04-09_fitto-joint_gaze-false_niter-10.csv'];
opts = detectImportOptions(model_results_file);
opts = setvartype(opts,{'sub_ind','SSE','BIC','LL'},'single');
opts = setvartype(opts,{'sub_ID','comb'},'string');
clear model_results
model_results = readtable(model_results_file,opts,'ReadRowNames',false);

% Which model function?
model_function = @cost_function_featureLearner_6;

% Which features to select from?
features = 'CoGrRiNa';
[basis_set,combs] = define_basis_set_2(features);

%% Preallocate
out_all = {};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GATHER DATA FOR GROUP LEVEL PCC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Settings
exclude = 201 + [0, 15, 22, 41, 119];
sub_indices = 201:353;
sub_indices = setdiff(sub_indices,exclude);
model_type = 'best';
asymmetric_LR = 0;
bounded_weights = 0;

% Run
store_participant_predictions = struct;
store_model_predictions = struct;
simulated_data = [];
sub_counter = 0;
SSEs = nan(length(sub_indices),1);
weights = nan(length(sub_indices),4);
for si = 1:length(sub_indices)
    sub_index = sub_indices(si);
    
    if mod(si,10) == 0
        disp('****************');
        disp(sub_index);
        disp('****************');
    end
    
    % Game data
    sub_ID = sub_IDs(sub_index);
    fprintf('Sub index: %i, sub ID: %s\n',sub_index, sub_ID);
    sub_dat = game_dat(game_dat.subID==sub_ID,:);

    try
        % Model data
        if strcmp(model_type, 'best')
            BICs = model_results{model_results{:,'sub_ind'}==sub_index,'BIC'};
            min_BIC = min(BICs);
            model_use = model_results(model_results{:,'sub_ind'}==sub_index ...
                & model_results{:,'BIC'} == min_BIC,:);
            comb = char(model_use.comb);
        elseif strcmp(model_type, 'full')
            comb = '1111';
            model_use = model_results(model_results{:,'sub_ind'}==sub_index ...
                & model_results{:,'comb'} == comb,:);
        end
        params = [model_use{1,'LR_up'}, model_use{1,'inv_temp'}];
        for i = 1:sum(comb=='1')
            params = [params model_use{1,sprintf('feature_weight_%i',i)}];
        end
        basis_sub_set = basis_set(comb=='1');
        basis_sub_set_names = strip(sprintf('%s_',basis_sub_set.name),'_');
        
        % Run model
        model_function = @cost_function_featureLearner_6;
        objective_function = @(params)model_function(params, ... % Free parameters
            sub_ID, sub_dat, [], ... % Input data
            basis_sub_set, 0, 0, 'joint', ... % Model structure
            0, 0, 1);
        out = objective_function(params');
        
        weights(sub_index,1:length(out.input.params(3:end))) = out.input.params(3:end);
        
        sub_counter = sub_counter + 1;
        
        % Get block-wise predictions & confidence
        for block = 1:4

            % Data
            block_dat = sub_dat(sub_dat.Block==(block-1),...
                {'Trial','S','T','CorrAns','GivenAns','Confidence','Type_Total'});
            player_type = char(unique(block_dat.Type_Total));

            % True & predicted
            player_choices = zeros(4,4);
            for ti = 1:16
                S = block_dat.S(ti); T = block_dat.T(ti);
                player_choices(Ss == S, Ts == T) = strcmp(block_dat.CorrAns{ti},'coop'); % Write as -1, 1
            end

            % Store
            store_participant_predictions.(player_type)(:,:,si) = ...
                out.participant_prediction_confidence_joint_grid_per_block.(sprintf('block_%i',block));
            store_model_predictions.(player_type)(:,:,si) = ...
                out.model_coop_likelihoods_per_block.(sprintf('block_%i',block));
        end
        sim_dat = sub_dat;
        sim_dat.model_pred_coop = out.store_pred_dist(:,1);
        writetable(sim_dat, sprintf('PPCs/simulations/simulations_sub-%i_subID-%s_%s_%s.csv',...
            sub_index, sub_ID, features, model_type));
        SSEs(si) = out.SSE;
        
    catch
        fprintf('Skipped subject index %i\n',sub_index);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOT GROUP LEVEL PCC
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define colormap
hex = (['#B2182B'; '#D6604D'; '#F4A582'; '#FDDBC7'; '#FFFFFF'; '#D1E5F0'; '#92C5DE'; '#4393C3'; '#2166AC']);
vec = (100:-(100/(length(hex)-1)):0)';
raw = sscanf(hex','#%2x%2x%2x',[3,size(hex,1)]).' / 255;
N = 128;
cmap = interp1(vec,raw,linspace(100,0,N),'pchip');
% cmap = csvread('~/Dropbox (Brown)/Postdoc FHL/JEROEN/SOC_STRUCT_LEARN/RdBu_viaPurp.csv');

% Plot
fig = figure;
colormap(cmap(end:-1:1,:));
player_types = {'trust_nat','opt_nat','pess_nat','env_nat'};
pt_fancy = {'Cooperative','Greedy','Risk-Averse','Envious'};
nrows = 3; ncols = length(player_types);
store_player_choices = struct;
for pti = 1:length(player_types)
    
    player_type = player_types{pti};
    block_dat = sub_dat(sub_dat.Type_Total==player_type,...
            {'Trial','S','T','CorrAns','GivenAns','Confidence','Type_Total'});
    player_choices = zeros(4,4);
    for ti = 1:16
        S = block_dat.S(ti); T = block_dat.T(ti);
        player_choices(Ss == S, Ts == T) = strcmp(block_dat.CorrAns{ti},'coop'); % Write as -1, 1
    end
    store_player_choices.(player_type) = player_choices;
    
    subplot(nrows,ncols,pti);
    imagesc('CData',player_choices);
    caxis([0,1]);
    xlabel('T'); ylabel('S');
    xticks(1:11); yticks(1:11);
    xticklabels(Ts); yticklabels(Ss);
    player_type_str =[player_type(1:find(player_type=='_')-1) '\_' player_type(find(player_type=='_')+1:end)];
    title(pt_fancy{pti});
    axis equal;
    axis tight;
    colorbar;

    subplot(nrows,ncols,ncols + pti);
    imagesc('CData',mean(store_participant_predictions.(player_type),3));
    caxis([0,1]);
    xlabel('T'); ylabel('S');
    xticks(1:11); yticks(1:11);
    xticklabels(Ts); yticklabels(Ss);
    title('Subject data');
    axis equal;
    axis tight;
    colorbar;
    
    subplot(nrows,ncols,ncols*2 + pti);
    imagesc('CData',mean(store_model_predictions.(player_type),3));
    caxis([0,1]);
    xlabel('T'); ylabel('S');
    xticks(1:11); yticks(1:11);
    xticklabels(Ts); yticklabels(Ss);
    title('Model');
    axis equal;
    axis tight;
    colorbar;

end

% all_mean_subjects = zeros(4,4);
% all_mean_model = zeros(4,4);
% all_mean_player = zeros(4,4);
% for pti = 1:length(player_types)
%     player_type = player_types{pti};
%     all_mean_player = all_mean_player + store_player_choices.(player_type) * 0.25;
%     all_mean_subjects = all_mean_subjects + mean(store_participant_predictions.(player_type),3) * .25;
%     all_mean_model = all_mean_model + mean(store_model_predictions.(player_type),3) * .25;
% end
% 
% subplot(nrows,ncols,5);
% imagesc('CData', all_mean_player);
% caxis([0,1]);
% xlabel('T'); ylabel('S');
% xticks(1:11); yticks(1:11);
% xticklabels(Ts); yticklabels(Ss);
% title('ALL player choices - mean');
% axis equal;
% axis tight;
% colorbar;
% 
% subplot(nrows,ncols,ncols + 5);
% imagesc('CData',all_mean_subjects);
% caxis([0,1]);
% xlabel('T'); ylabel('S');
% xticks(1:11); yticks(1:11);
% xticklabels(Ts); yticklabels(Ss);
% title('Subject data');
% axis equal;
% axis tight;
% colorbar;
% 
% subplot(nrows,ncols,ncols*2 + 5);
% imagesc('CData',all_mean_model);
% caxis([0,1]);
% xlabel('T'); ylabel('S');
% xticks(1:11); yticks(1:11);
% xticklabels(Ts); yticklabels(Ss);
% title(sprintf('Model %s',comb));
% axis equal;
% axis tight;
% colorbar;


saveas(fig, sprintf('PPCs/PPC_study4_%s.eps',features), 'epsc');





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOT PCC WITHOUT ANY TEXT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define colormap
hex = flipud(['#B2182B'; '#D6604D'; '#F4A582'; '#FDDBC7'; '#FFFFFF'; '#D1E5F0'; '#92C5DE'; '#4393C3'; '#2166AC']);
vec = (100:-(100/(length(hex)-1)):0)';
raw = sscanf(hex','#%2x%2x%2x',[3,size(hex,1)]).' / 255;
N = 128;
cmap = interp1(vec,raw,linspace(100,0,N),'pchip');

% Plot
fig = figure;
colormap(cmap);
player_types = {'opt_nat','pess_nat','opt_inv','pess_inv'};
nrows = 3; ncols = length(player_types);
store_player_choices = struct;
for pti = 1:length(player_types)
    
    player_type = player_types{pti};
    block_dat = sub_dat(sub_dat.Type_Total==player_type,...
            {'Trial','S','T','CorrAns','GivenAns','Confidence','Type_Total'});
    player_choices = zeros(4,4);
    for ti = 1:16
        S = block_dat.S(ti); T = block_dat.T(ti);
        player_choices(Ss == S, Ts == T) = strcmp(block_dat.CorrAns{ti},'coop'); % Write as -1, 1
    end
    store_player_choices.(player_type) = player_choices;
    
    subplot(nrows,ncols,pti);
    imagesc('CData',player_choices);
    caxis([0,1]);
    axis equal;
    axis tight;
    axis off
    xticks([]); yticks([]); xlabel([]); ylabel([]);

    subplot(nrows,ncols,ncols + pti);
    imagesc('CData',mean(store_participant_predictions.(player_type),3));
    caxis([0,1]);
    axis equal;
    axis tight;
    axis off
    xticks([]); yticks([]); xlabel([]); ylabel([]);
    
    subplot(nrows,ncols,ncols*2 + pti);
    imagesc('CData',mean(store_model_predictions.(player_type),3));
    caxis([0,1]);
    axis equal tight off;
%     axis tight;
    xticks([]); yticks([]); xlabel([]); ylabel([]);
    
end

tightfig;

% saveas(fig, sprintf('~/Desktop/PPC_%s_tight.png',features));
export_fig(fig, sprintf('~/Desktop/PPC_tight_%s_%s.png',features,model_type), '-transparent')

%%
fig_c = figure;
colormap(cmap);
% imagesc('CData',mean(store_participant_predictions.(player_type),3));
caxis([0,1]);
cbax = colorbar('XTick',[]);
axis off;
export_fig(fig_c, '~/Desktop/PPC_colorbar.png', '-transparent');

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

function model_results = load_model_results(study_name, model_date_string, fit_to, features, niter)

    suffix = '';
    if ~isempty(fit_to)
        suffix = [suffix sprintf('_fitto-%s',fit_to)];
    end
    if ~isempty(features)
        suffix = [suffix sprintf('_features-%s',features)];
    end
    if ~isempty(niter)
        suffix = [suffix sprintf('_niter-%i',niter)];
    end
    base_dir = '/Users/jvanbaar/Dropbox (Brown)/Postdoc FHL/JEROEN/SOC_STRUCT_LEARN/';
    model_results_fname = [base_dir, sprintf('%s/Data/Cleaned/',study_name), ...
        sprintf('model_results_%s%s.csv',model_date_string, suffix)];
    fprintf('Attempting to load model results from:\n%s.\n',model_results_fname);
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
        if strcmp(features_split{fi},'InverseGreed')
            abbreviation = 'Ig';
        elseif strcmp(features_split{fi}, 'InverseRisk')
            abbreviation = 'Ir';
        else
            abbreviation = features_split{fi}(1:2);
        end
        features = [features, abbreviation];
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
    out = cost_function_featureLearner_6(params, ... % Free parameters
        sub_ID, sub_dat, [], ... % Input data
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
%             keyboard
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