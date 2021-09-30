%% POSTERIOR PREDICTIVE CHECKS - GAZE
%
% -------------------------------------------------------------------------
% Goal: Predict gaze trial-by-trial using 'value of information' computation
% -------------------------------------------------------------------------
%
% Steps:
% 1. Select best model per subject
% 2. Run model again, this time with gaze set to true
% 3. Record predicted value of information predicted by best model
% 4. Store

%% Basic definitions

%% Load functions
addpath('helpers');

%% Game variables; softmax choice rule
define_globals;

%% Load behavioral data
[game_dat, sub_IDs] = load_game_dat;

%% Load gaze data (used as input to function but does not affect gaze prediction)
[gaze_dat, ~] = load_gaze_dat;

%% Load model results
bestpersubsfile = ['/Users/jeroen/Dropbox (Brown)/Postdoc FHL/JEROEN/SOC_STRUCT_LEARN/',...
    'Study2_EyeTracking/Data/Cleaned/Model_results/Best_motives_per_participant.csv'];
opts = detectImportOptions(bestpersubsfile);
opts = setvartype(opts,{'sub_ind','SSE','BIC','LL'},'single');
opts = setvartype(opts,{'sub_ID','comb'},'string');
clear model_results
model_results = readtable(bestpersubsfile,opts,'ReadRowNames',false);

% Which model function?
model_function = @cost_function_featureLearner_6;

% Which features to select from?
[basis_set,combs] = define_basis_set_2('CoGrRiNa');

%% Define target directory
out_dir = 'Gaze_predict';

%% Select subject
for sub_index = 151:200
    disp(sub_index);
    % Behavioral data
    sub_ID = sub_IDs(sub_index);
    fprintf('Game data loaded with sub ID: %s\n',sub_ID);
    sub_dat = game_dat(game_dat.subID==sub_ID,:);

    % Gaze data
    sub_gaze_dat = gaze_dat(gaze_dat.sub == str2double(sub_ID) - 5000,:);
    trials = unique(sub_gaze_dat.trial)';
    if isempty(trials)
        error('No gaze data present for subject %s.\n',sub_ID);
    else
        fprintf('Loading gaze data of %i trials.\n', length(trials));
    end
    trial_gaze_diffs = nan(length(trials),1);
    relative_gazes = nan(length(trials),2);

    for trial = 1:128
        if isempty(find(trials==trial,1))
            trial_gaze_diffs(trial) = 0;
            relative_gazes(trial,:) = [0,0];
        else
            trial_gaze_dat = sub_gaze_dat(sub_gaze_dat.trial == trial,...
                {'trial','num_S_T','dur_pct'});
            % Edit this to bring in line with Stojic relative fixation
            S_actual = sum(trial_gaze_dat{strcmp(trial_gaze_dat.num_S_T, "S"),'dur_pct'});
            T_actual = sum(trial_gaze_dat{strcmp(trial_gaze_dat.num_S_T, "T"),'dur_pct'});
            if S_actual == 0 && T_actual == 0
                gaze_diff = 0;
                relative_gazes(trial,:) = [0,0];
            else
                gaze_diff = (S_actual - T_actual) / (S_actual + T_actual);
                relative_gazes(trial,:) = ...
                    [S_actual / (S_actual + T_actual), T_actual / (S_actual + T_actual)];
            end
            trial_gaze_diffs(trial) = gaze_diff;

        end
    end
    trial_gaze_diffs(isnan(trial_gaze_diffs)) = 0;

    % Model
    sub_model = model_results(model_results{:,'sub_ind'}==sub_index,:);
    comb = char(sub_model{1,'comb'});
    params = [sub_model{1,'LR_up'}, sub_model{1,'inv_temp'}, 0]; % The 0 is the gaze bias
    for i = 1:sum(comb=='1')
        params = [params sub_model{1,sprintf('feature_weight_%i',i)}];
    end
    basis_sub_set = basis_set(comb=='1');
    basis_sub_set_names = strip(sprintf('%s_',basis_sub_set.name),'_');

    % Define output
    curdir = pwd;
    filename = sprintf('%s/%s/KL_gaze_predictions_sub-%i.csv', curdir, out_dir, sub_index);
    if exist(filename,'file') == 2
        delete(filename);
        fprintf('Deleted existing data file %s\n',filename);
    end
    KL_gaze_predictions_fid = fopen(filename,'a');
    fprintf(KL_gaze_predictions_fid, 'subInd,subID,trial,KL_S,KL_T,diff_T_S\n');

    % Run model
    model_out = model_function(params', ... % Free parameters
                    sub_ID, sub_dat, relative_gazes, ... % Input data
                    basis_sub_set, 0, 0, 'joint', ... % Model structure
                    1, 0, true);
    
    % Write KL-based gaze predictions
    for ti = 1:128
        fprintf(KL_gaze_predictions_fid, '%i,%s,%i,%.6f,%.6f,%.6f\n',...
            sub_index,sub_ID,ti,model_out.info_values(ti,1),model_out.info_values(ti,2),...
            model_out.info_values(ti,2) - model_out.info_values(ti,1));
    end

end