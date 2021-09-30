%% v4
% Fits to choice and confidence data.

function feature_learner_fit_4(sub_index, n_iter, varargin)
    
    % -------------------------------------------------------------------------
    % Inputs to constrain model types considered:
    % -------------------------------------------------------------------------
    %
    % ----- REQUIRED inputs -----
    % 'fit_to':          Followed by 'choice_only' or 'joint'. 'joint' means we transform choice and
    %                    confidence into 1 joint measure, where 0 means 100% confidence for
    %                    Defect, 1 means 100% confidence for Cooperate, and 0.5 is 0% confidence
    %                    for either.
    % 'features':        Followed by basis set definition. Can be e.g.
    %                    'CoGrRiReNa' which includes Coop, Greed, Risk,
    %                    Regret, and Nash.
    %
    % ----- OPTIONAL inputs -----
    % 'comb_ind':        Followed by feature combination index. The
    %                    numbering depends on the number of features in the set.
    %                    Can also be 'fullmodel' to only include the model
    %                    with all bases.
    % 'asymmetric_LR':   Followed by 1 (true) or 0 (false). If not
    %                    specified, both variants are run.
    % 'bounded_weights': Followed by 1 (true) or 0 (false). If not
    %                    specified, both variants are run.
    % 'gaze':            Followed by 1 (true) or 0 (false). If not
    %                    specified, gaze is NOT used as input to the model.
    % 
    % Examples:
    % varargin = {'fit_to','joint','features','CoGrRiReNaEn'}
    % varargin = {'fit_to','joint','features','CoGrRiNa','bounded_weights',0,'asymmetric_LR',0,'gaze',1}
    % varargin = {'fit_to','joint','features','CoHgShSgPd','comb_ind','fullmodel','bounded_weights',0,'asymmetric_LR',0,'gaze',0}
    % 

    %% Default arguments
    
    % Asymmetric learning rate, bounded weights, comb index input
    asymmetric_LR_list = [false, true];
    bounded_weights_list = [false, true];
    comb_index_choice = 0;
    gaze = false;

    %% Parse variable arguments
    
    if isempty(varargin) || (sum(strcmp(varargin,'features')) + sum(strcmp(varargin,'fit_to')) < 2)
        error('2 required inputs: fit_to, features');
    end
    for varg = 1:length(varargin)
        if ischar(varargin{varg})
            if strcmpi('fit_to',varargin{varg})
                fit_to = varargin{varg + 1};
                fprintf('Found fit_to: %s\n',fit_to);
                varargin{varg} = {}; varargin{varg + 1} = {};
            end
            if strcmpi('features',varargin{varg})
                features = varargin{varg + 1};
                fprintf('Features in basis set: %s\n',features);
                varargin{varg} = {}; varargin{varg + 1} = {};
            end
            if strcmpi('comb_ind',varargin{varg})
%                 if isnumeric(varargin{varg + 1})
                comb_index_choice = varargin{varg + 1};
%                 else
%                     
%                 end
                varargin{varg} = {}; varargin{varg + 1} = {};
            end
            if strcmpi('asymmetric_LR',varargin{varg})
                asymmetric_LR_list = logical(varargin{varg + 1});
                varargin{varg} = {}; varargin{varg + 1} = {};
            end
            if strcmpi('bounded_weights',varargin{varg})
                bounded_weights_list = logical(varargin{varg + 1});
                varargin{varg} = {}; varargin{varg + 1} = {};
            end
            if strcmpi('gaze',varargin{varg})
                gaze = logical(varargin{varg + 1});
                varargin{varg} = {}; varargin{varg + 1} = {};
            end
        end
    end
    
    %% Load functions
    addpath('helpers');

    %% Game variables; softmax choice rule
    define_globals;

    %% Define basis set
    [basis_set,combs] = define_basis_set_2(features);
    n_features_full_basis_set = length(basis_set);
    
    % Select only 1 feature combination?
    comb_index_list = 1:length(combs);
    if isnumeric(comb_index_choice)
        if comb_index_choice > 0 % This means one comb index was input by user
            comb_index_list = comb_index_choice;
        end
    elseif strcmp(comb_index_choice, 'fullmodel')
        comb_index_list = comb_index_list(end);
    end
    
    %% Load data
    [game_dat, sub_IDs] = load_game_dat;
    sub_ID = sub_IDs(sub_index);
    sub_dat = game_dat(game_dat.subID==sub_ID,:);
    fprintf('Game data loaded with sub ID: %s\n',sub_ID);
    
    %% Load gaze data
    if gaze
        
        [gaze_dat, ~] = load_gaze_dat;
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
    else
        fprintf('No gaze data loaded.\n');
        trial_gaze_diffs = [];
        relative_gazes = [];
    end
    
    %% Open log / results files
% 
    tmp = what('../../../..');
    data_dir = [tmp.path, '/data/jvanbaar/SOC_STRUCT_LEARN/ComputationalModel/FeatureRL'];
    results_dir = sprintf('%s/Results/results_%s',data_dir,datestr(now,'yyyy-mm-dd'));
    % model_dir = what('..');
    % model_dir = model_dir.path;
    % results_dir = sprintf('%s/Results/results_%s',model_dir,datestr(now,'yyyy-mm-dd'));
    fprintf('Using results directory %s\n',results_dir);
    if ~exist(results_dir,'dir')
        mkdir(results_dir);
        fprintf('Created results directory %s\n',results_dir);
    end
    log_filename_base = sprintf('%s/subInd-%i_subID-%s_fitto-%s_features-%s_gaze-%s_niter-%i_log.csv',...
        results_dir, sub_index, sub_ID, fit_to, features, string(gaze), n_iter);
    filename = log_filename_base;
    suffix = 1;
    while exist(filename,'file') == 2
        suffix = suffix + 1;
        filename = [log_filename_base(1:end-4) sprintf('_%i.csv',suffix)];
    end
    log_ID = fopen(log_filename_base,'a');
    header_line = ['sub_ind,sub_ID,fit_to,comb_index,comb,feature_names,asymm_LR,bounded_weights,gaze,', ...
        'LR_up,LR_down,inv_temp,gaze_bias,' sprintf('feature_weight_%i,',1:n_features_full_basis_set) 'cost_type,SSE,LL,BIC\n'];
    fprintf(log_ID, header_line);
    
    %% Fit models
    n_fail = 0;
    visualize = false; returnInfo = false;
    clear options
    options = optimoptions('fmincon');
%     options.Display = 'iter';
    out = {};
    
    fprintf('Running the following model versions:\n');
    disp(fit_to);
    disp(comb_index_list);
    disp(asymmetric_LR_list);
    disp(bounded_weights_list);
    disp(gaze);
    
    if strcmp(fit_to, 'joint')
        cost_type = 'SSE';
    elseif strcmp(fit_to, 'choice_only')
        cost_type = 'NLL';
    end
    
    for comb_index = comb_index_list
        comb = combs(comb_index,:);
        basis_sub_set = basis_set(comb=='1');
        basis_sub_set_names = strip(sprintf('%s_',basis_sub_set.name),'_');
        
        for asymmetric_LR = asymmetric_LR_list
            for bounded_weights = bounded_weights_list
                fprintf(['Fitting featureRL model to %s, comb %s, gaze %s, asymmetric LR %s, ',...
                    'bounded weights %s. %s\n'], ...
                    fit_to, comb, string(gaze), string(asymmetric_LR), ...
                    string(bounded_weights), datetime);

                % Parameter bounds
                param_bounds = [repmat([0.01   , 10],(1 + double(asymmetric_LR)),1); ... % LR
                               0.01   , 5; ... % InvTemp
                               repmat([0,10],double(gaze),1); ... % Gaze bias
                               repmat([0 - (1 - double(bounded_weights)) * 20,20],...
                                    length(basis_sub_set),1); ... % Weights
                               ];
                lb = param_bounds(:,1);
                ub = param_bounds(:,2);

                % Define objective function
                model_function = @cost_function_featureLearner_6;
                objective_function = @(params)model_function(params, ... % Free parameters
                    sub_ID, sub_dat, relative_gazes, ... % Input data
                    basis_sub_set, bounded_weights, asymmetric_LR, fit_to, ... % Model structure
                    gaze, visualize, returnInfo);

                % Prepare output storage
                iter_out = nan(n_iter, 10);

                % Fit model
                for n = 1:n_iter

                    notdone = 1;
                    while notdone
                        try
                            % Fit
                            params0 = lb + rand(size(param_bounds,1),1).*(ub-lb);
                            [x,cost] = fmincon(objective_function, params0, ...
                                [], [], [], [], lb, ub, [], options);
                            notdone = 0;
                            
                            % Store iteration
                            iter_out(n,1) = cost;
                            iter_out(n,2:(1+length(x))) = x';
                            disp([n, cost, x']);
                        catch
                            n_fail = n_fail + 1;
                            fprintf('Failed %i times.\n',n_fail);
                            if n_fail > 100
                                notdone = 0;
                            end
                        end
                    end
                    
                end

                % Extract
                [best_cost, best_iter] = min(iter_out(:,1));
                best_params = iter_out(best_iter,2:end);
                if asymmetric_LR
                    LR_up = best_params(1);
                    LR_down = best_params(2);
                    inv_temp = best_params(3);
                    n_learning_params = 3;
                else
                    LR_up = best_params(1);
                    LR_down = best_params(1);
                    inv_temp = best_params(2);
                    n_learning_params = 2;
                end
                if gaze
                    gaze_bias = best_params(n_learning_params + 1);
                    feature_weights = best_params(n_learning_params + 2:n_learning_params + 1 + n_features_full_basis_set);
                else
                    feature_weights = best_params(n_learning_params + 1:n_learning_params + n_features_full_basis_set);
                    gaze_bias = 0;
                end
                
                % Run best model to get NLL and BIC
                model_out = model_function(best_params(1:length(x))', ... % Free parameters
                    sub_ID, sub_dat, relative_gazes, ... % Input data
                    basis_sub_set, bounded_weights, asymmetric_LR, fit_to, ... % Model structure
                    gaze, visualize, true);
                if strcmp(fit_to,'joint')
                    SSE = model_out.SSE;
                    if best_cost ~= SSE
                        error('Something went wrong: SSE not consistent across model runs');
                    end
                else
                    SSE = -999;
                end
                LL = -model_out.NLL;
                BIC = model_out.BIC;
                
                % Store
                fprintf(log_ID, ['%i,%s,%s,%i,%s,%s,%i,%i,%i,',... % Sub / model info
                                    '%.3f,%.3f,%.3f,%.3f,',... % LRs and inv temp and gaze bias
                                    repmat('%.3f,',1,n_features_full_basis_set),... % Feature weights
                                    '%s,%.3f,%.3f,%.3f\n'],... % SSE LL BIC
                    sub_index, sub_ID, fit_to, comb_index, comb, basis_sub_set_names, ...
                    double(asymmetric_LR), double(bounded_weights),double(gaze),...
                    LR_up, LR_down, inv_temp, gaze_bias,...
                    feature_weights,...
                    cost_type, SSE, LL, BIC);
                to_append = {sub_index, sub_ID, fit_to, comb_index, comb, basis_sub_set_names, ...
                    double(asymmetric_LR), double(bounded_weights),double(gaze),...
                    LR_up, LR_down, inv_temp, gaze_bias,...
                    feature_weights,...
                    cost_type, SSE, LL, BIC};
                out = [out; to_append];
            end
        end
    end
    
    % Store results
    filename_base = sprintf('%s/subInd-%i_subID-%s_fitto-%s_features-%s_gaze-%s_niter-%i_results.mat',...
        results_dir, sub_index, sub_ID, fit_to, features, string(gaze), n_iter);
    filename = filename_base;
    suffix = 1;
    while exist(filename,'file') == 2
        suffix = suffix + 1;
        filename = [filename_base(1:end-4) sprintf('_%i.mat',suffix)];
    end
    save(filename,'out');
    fprintf('Done.\n');
    
end
