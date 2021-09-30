%% POSTERIOR PREDICTIVE CHECKS FOR MODEL
%
% -------------------------------------------------------------------------
% Goals:
% -------------------------------------------------------------------------
%
% 3. Plot & store PPCs as learning curves for clusters of subjects with and
%       without particular considered motives.
% Matt: "Maybe the first step would be to view the learning dynamics. Have 
%        you tried plotting accuracy across trials as a function of the 
%        simulated participant and best fitting model type? The idea would
%        be to use model selection to cluster participants -- then to see 
%        whether you can capture the learning dynamics of each cluster -- 
%        which is a sort of happy medium between the group average and each
%        particpant individually."
% The function of this analysis is to eventually see whether there is a lot
% of room for the model to improve or not.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load functions
addpath('helpers');
% Game variables; softmax choice rule
define_globals;
% Load data
[game_dat, cfg.sub_IDs] = load_game_dat;
% Load model results
study_name = 'Study2_EyeTracking';
model_date_string = '2019-10-28';
fit_to = 'joint';
features = 'CoGrRiReNaEn';
niter = 5;
model_results = load_model_results(study_name, model_date_string, fit_to, features, niter);
% Preallocate
out_all = {};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GATHER DATA FOR GROUP LEVEL PCC - INCL LEARNING CURVES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Settings
sub_indices = 1:50;
comb = {'100000','010000','001000','000010',...
        '110000','101000','100010','011000','010010','001010',...
        '111000','110010','101010','011010',...
        '111010'}; % Yields each subject's best model while only considering CoGrRiNa - '111010' would give CoGrRiNa
asymmetric_LR = 0;
bounded_weights = [];

% Run
store_participant_predictions = struct;
store_model_predictions = struct;
store_participant_lc = struct;
store_model_lc = struct;
store_model_use = table;
n_perms = 100;
sub_counter = 0;
for sub_index = sub_indices
    if strcmp(study_name, 'Study2_EyeTracking')
        sub_ind_use = sub_index + 150;
    end
    
    if mod(sub_index,10) == 0
        disp('****************');
        disp(sub_index);
        disp('****************');
    end
    
    % Game data
    sub_ID = cfg.sub_IDs(sub_ind_use);
    fprintf('Sub index: %i, sub ID: %s\n',sub_ind_use, sub_ID);
    sub_dat = game_dat(game_dat.subID==sub_ID,:);

    try
        % Model data
        model_use = select_model(model_results, sub_ind_use, comb, asymmetric_LR, bounded_weights);
        store_model_use = [store_model_use; model_use];
        
        if sub_counter == 0 % Do this only once
            features_string = model_use.feature_names{1};
            features_split = strsplit(features_string,'_');
            n_features = length(features_split);
            features_disp = '';
            for fi = 1:n_features
                features_disp = [features_disp, features_split{fi}(1:2)];
            end
            disp(features_disp);
            pause(.3);
        end
        
        % Run model
        [out, ~] = run_model(sub_ID, sub_dat, model_use, []);

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
            store_participant_predictions.(player_type)(:,:,sub_index) = ...
                out.participant_prediction_confidence_joint_grid_per_block.(sprintf('block_%i',block));
            store_model_predictions.(player_type)(:,:,sub_index) = ...
                out.model_coop_likelihoods_per_block.(sprintf('block_%i',block));
            
            % Learning curve - model
            lc_model = nan(16,n_perms);
            for i = 1:n_perms
                preds = rand(16,1) <= out.store_pred_dist(1 + (block-1) * 16 : 16 + (block-1) * 16,1);
                lc_model(:,i) = cumsum(strcmp(block_dat.CorrAns,'coop') == preds);
            end
            lc_model_mean = mean(lc_model, 2);
%             model_lc_range = [mean(lc_model') - std(lc_model'), fliplr(mean(lc_model') + std(lc_model'))];
            % Learning curve - true
            lc_participant = cumsum(block_dat.CorrAns == block_dat.GivenAns);
            
            % Store learning curves
            store_participant_lc.(player_type)(:,:,sub_index) = lc_participant;
            store_model_lc.(player_type)(:,:,sub_index) = lc_model_mean;
            
        end
    catch
        fprintf('Skipped subject index %i\n',sub_ind_use);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOT GROUP LEVEL PCC AS LEARNING CURVES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig = figure;
player_types = {'opt_nat','pess_nat','opt_inv','pess_inv'};
nrows = 1; ncols = length(player_types);
for pti = 1:length(player_types)
    
    player_type = player_types{pti};
    
    participant_lc_mean = mean(store_participant_lc.(player_type),3);
    participant_lc_sterr = std(store_participant_lc.(player_type),0,3)/sqrt(length(sub_indices));
    model_lc_mean = mean(store_model_lc.(player_type),3);
    model_lc_sterr = std(store_model_lc.(player_type),0,3)/sqrt(length(sub_indices));
    
    subplot(nrows,ncols,pti);
    hold on;
    errorbar(1:16, participant_lc_mean, participant_lc_sterr, 'b', 'LineWidth', 2);
    errorbar(1:16, model_lc_mean, model_lc_sterr, 'b:', 'LineWidth', 2);
    plot(1:16, .5:.5:8, 'k:');
    xlim([1,16]); ylim([0,16]);
    ylabel('Cumulative score'); xlabel('Trial');
    legend({'Participant data (mean ± sterr)','Best model pp (mean ± sterr)','Chance'});
    title(replace(player_type,'_','\_'));

end

% saveas(fig, sprintf('~/Desktop/PPC_learning-curve_%s.png',features));





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PLOT CLUSTER LEVEL PCC AS LEARNING CURVES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fig = figure;
motive_clusters = {'Coop','Greed','Risk','Nash'};
player_types = {'opt_nat','pess_nat','opt_inv','pess_inv'};
nrows = length(motive_clusters); ncols = length(player_types);

for ci = 1:length(motive_clusters)
    
    haves = logical(contains(store_model_use.feature_names, motive_clusters{ci}));
    havenots = logical(1 - haves);

    for pti = 1:length(player_types)

        player_type = player_types{pti};

        haves_participant_lc_mean = mean(store_participant_lc.(player_type)(:,:,haves),3);
        haves_participant_lc_sterr = std(store_participant_lc.(player_type)(:,:,haves),0,3)/sqrt(length(sub_indices));
        haves_model_lc_mean = mean(store_model_lc.(player_type)(:,:,haves),3);
        haves_model_lc_sterr = std(store_model_lc.(player_type)(:,:,haves),0,3)/sqrt(length(sub_indices));
        havenots_participant_lc_mean = mean(store_participant_lc.(player_type)(:,:,havenots),3);
        havenots_participant_lc_sterr = std(store_participant_lc.(player_type)(:,:,havenots),0,3)/sqrt(length(sub_indices));
        havenots_model_lc_mean = mean(store_model_lc.(player_type)(:,:,havenots),3);
        havenots_model_lc_sterr = std(store_model_lc.(player_type)(:,:,havenots),0,3)/sqrt(length(sub_indices));

        subplot(nrows,ncols,pti + (ci-1)*ncols);
        hold on;
        
        errorbar(1:16, haves_participant_lc_mean, haves_participant_lc_sterr, 'b', 'LineWidth', 2);
        errorbar(1:16, haves_model_lc_mean, haves_model_lc_sterr, 'm', 'LineWidth', 2);
        errorbar(1:16, havenots_participant_lc_mean, havenots_participant_lc_sterr, 'b:', 'LineWidth', 2);
        errorbar(1:16, havenots_model_lc_mean, havenots_model_lc_sterr, 'm:', 'LineWidth', 2);
        plot(1:16, .5:.5:8, 'k:');
        xlim([1,16]); ylim([0,16]);
        ylabel('Cumulative score'); xlabel('Trial');
        legend({sprintf('With %s',motive_clusters{ci}),'Model',...
            'Without','Model'},'Location','NorthWest');
        title([motive_clusters{ci}, ' - ', replace(player_type,'_','\_')]);

    end
    
end

% saveas(fig, sprintf('~/Desktop/PPC_learning-curve_%s_cluster-by-motive.png',features));





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
        if ischar(comb)
            subject_model_results = subject_model_results(...
                  strcmp(subject_model_results.comb,comb),:);
            fprintf('Using custom comb: %s.\n',comb);
        elseif iscell(comb)
            model_rows = [];
            for combi = comb
                model_rows = [model_rows, strcmp(subject_model_results.comb,combi)]; %#ok<AGROW>
            end
            model_rows = logical(sum(model_rows,2));
            subject_model_results = subject_model_results(model_rows,:);
            fprintf(['Allowing combs: %s',repmat(', %s',1,length(comb)-1),'.\n'],comb{:});
        end
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