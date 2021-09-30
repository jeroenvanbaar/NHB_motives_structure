%% Version history:
% Added asymmetric learning rate
% Added fit to both behavior and confidence
% 10/15/19: added support for random basis function (which have field
% 'grid' instead of 'model' and require S/T grid index instead of S/T value
% 10/21/19: Switched to SSE as cost function (output value). 'joint_joey_style' is now
% simply called 'joint', the old 'joint' option is discontinued. Cleaned
% up.
% 12/03/19: Added relative T/S looking time to scale weight on Gr/Ri bases.
% 03/24/20: Implemented the option of having bases that do not inform all
% S/T pairs. That is, bases now contain 1s, 0s, and -1s, with 1 meaning
% coop, -1 meaning defect, and 0 meaning no prediction.
% 03/31/20: Changed the 'gaze' component of the model such that gaze no
% longer manipulates choice, but instead we only generate a predicted gaze
% per trial which we can compare to the true gaze data.

function out = cost_function_featureLearner_6(...
    params, ... % Free parameters
    sub_ID, sub_dat, relative_gazes, ... % Input data
    basis_sub_set, bounded_weights, asymmetric_LR, fit_to, ... % Model structure
    gaze, visualize, returnInfo) % Do you want to see the inner workings of the model?
    
    %%
    %%% SOME BASICS
    basis_sub_set_size = length(basis_sub_set);
    basis_sub_set_names = cell(basis_sub_set_size,1);
    for bfi = 1:basis_sub_set_size
        basis_sub_set_names{bfi} = basis_sub_set(bfi).name;
    end
    global Ss Ts softmax
    
    %%% PARSE FREE PARAMETERS
    if asymmetric_LR
        LRs = params(1:2)';
        n_learning_params = 3;
    else
        LRs = [params(1), params(1)];
        n_learning_params = 2;
    end
    inv_temp = params(n_learning_params);
    if gaze
        gaze_bias = params(n_learning_params + 1);
        initial_weights = params((n_learning_params + 2):(n_learning_params + 1 + basis_sub_set_size));
    else
        initial_weights = params((n_learning_params + 1):(n_learning_params + basis_sub_set_size));
    end
    n_params = sum(~isnan(params));
%     disp(initial_weights);
    %%% INITIALIZE
    if gaze
        info_values = nan(height(sub_dat),2);
    end
    store_coop_values = nan(height(sub_dat),1);
    store_pred_dist = nan(height(sub_dat),2);
    store_model_prediction = nan(height(sub_dat),1);
    store_participant_prediction_confidence_joint = nan(height(sub_dat),1);

    %%% VIZ
    if visualize
        fig = figure;
    end

    %% RUN
    blocks = unique(sub_dat.Block)';
    for bi = 1:length(blocks)

        %%% INITIALIZE
        player_choice_grid = zeros(4,4); % Initialize at 0
        participant_prediction_grid = zeros(4,4);
        participant_confidence_grid = zeros(4,4);
        participant_prediction_confidence_joint_grid = zeros(4,4);
        
        model_coop_likelihood_grid = 0.5*ones(4,4);
        weights = initial_weights;
        weights_updating = nan(17,basis_sub_set_size);
        weights_updating(1,:) = weights;
        if gaze
            weights_modulated_by_gaze_updating = nan(16,basis_sub_set_size);
        end
        block = blocks(bi);
        block_dat = sub_dat(sub_dat.Block==block,...
                    {'Trial','S','T','CorrAns','GivenAns','Confidence'});
        player_type = unique(sub_dat(sub_dat.Block==block, {'Type_Total'}));
        player_type = player_type.Type_Total;
        num_trials = height(block_dat);

        for ti = 1:num_trials

            trial = (bi-1)*16+ti;
            
            %%% GAME
            S = block_dat.S(ti);
            T = block_dat.T(ti);

            %%% INFORMATION SEARCH
            if gaze
                
                % % Matt's method => compute coop proba for all S and T, then
                % marginalize over S/T
                conditional_distribution = zeros(4,4);
                for S_hypothetical = Ss
                    for T_hypothetical = Ts
                        basis_function_choices = nan(basis_sub_set_size,1);
                        for bfi = 1:basis_sub_set_size
                            basis_function_choices(bfi) = ...
                                2*basis_sub_set(bfi).model(S_hypothetical,T_hypothetical) - 1;
                        end
                        hypothetical_likelihoods = softmax([sum(weights.*basis_function_choices), 0], inv_temp);
                        conditional_distribution(Ss == S_hypothetical, Ts == T_hypothetical) = hypothetical_likelihoods(1);
                    end
                end
                marginal_distribution_S = repmat(mean(conditional_distribution,1),4,1);
                marginal_distribution_T = repmat(mean(conditional_distribution,2),1,4);
                info_value_KL_S = kldiv(1:16,conditional_distribution(1:end)./sum(conditional_distribution(1:end)),...
                    marginal_distribution_S(1:end)./sum(marginal_distribution_S(1:end)));
                info_value_KL_T = kldiv(1:16,conditional_distribution(1:end)./sum(conditional_distribution(1:end)),...
                    marginal_distribution_T(1:end)./sum(marginal_distribution_T(1:end)));
                info_values((bi-1)*16+ti,:) = [info_value_KL_S, info_value_KL_T];
                info_value_diff = (info_value_KL_S - info_value_KL_T) / ...
                    max(0.00001,sum([info_value_KL_S, info_value_KL_T]));
                        % This is bounded between [-1,1]. 1 means the model
                        % thinks it's ONLY valuable to look at S, -1 is T
                        % The max() term takes care of the case in which
                        % neither S nor T has info value according to the
                        % model (divide by 0).
                
                % Load actual gaze dat
%                 gaze_diff = trial_gaze_diffs(block*16 + ti);
                         % This is also bounded between [-1,1]. 1 means the
                         % person ONLY looked at S (100% of the total time across the two), -1 ONLY at T
%                 temporary_bias_toward_S = (gaze_diff - info_value_diff) / 2;
                        % This should be bounded between [-1,1].
                relative_gaze = relative_gazes(trial,:);
                        % This is bounded between 0 and 1
                
                %%% VIZ
                if visualize
                    subplot(2,8,5:6);
                    bar([info_value_KL_S, info_value_KL_T]);
                    xticklabels({'S','T'}); ylabel({'KL divergence conditional -> marginal'});
                    title('Information value (subject)');
                end
            end

            %%% PREDICTION
            
            % Load participant's prediction
            participant_prediction = strcmp(block_dat.GivenAns{ti},'coop'); % Write as 0, 1
            participant_prediction_grid(Ss == S, Ts == T) = 2 * participant_prediction - 1; % Write as -1, 1
            participant_confidence = block_dat.Confidence(ti) / 100;
            participant_confidence_grid(Ss == S, Ts == T) = participant_confidence;
            if participant_prediction == 1
                participant_prediction_confidence_joint = 0.5 + 0.5 * participant_confidence;
            else
                participant_prediction_confidence_joint = 0.5 - 0.5 * participant_confidence;
            end
            participant_prediction_confidence_joint_grid(Ss == S, Ts == T) = participant_prediction_confidence_joint;
            store_participant_prediction_confidence_joint((bi-1)*16+ti) = participant_prediction_confidence_joint;
            
            % Make model predictions (probabilities)
            basis_function_choices = nan(basis_sub_set_size,1);
            for bfi = 1:basis_sub_set_size
                basis_function_choices(bfi) = basis_sub_set(bfi).model(S,T);
            end
            coop_value = sum(weights.*basis_function_choices); % estimated value
            store_coop_values((bi-1)*16+ti) = coop_value;
            
            %%% HERE WE WILL MODULATE THE WEIGHTS BY RELATIVE T/S LOOKING TIME
            if gaze
                greed_index = find(contains(basis_sub_set_names, 'Greed'), 1);
                risk_index = find(contains(basis_sub_set_names, 'Risk'), 1);
                weights_modulated_by_gaze = weights;
                if ~isempty(greed_index)
%                     weights_modulated_by_gaze(greed_index) = ...
%                         weights_modulated_by_gaze(greed_index) * (1 - max(min(gaze_bias*temporary_bias_toward_S,1),-1));
%                     weights_modulated_by_gaze(greed_index) = ...
%                         weights_modulated_by_gaze(greed_index) * (1 - gaze_bias*temporary_bias_toward_S);
                            % We want this weight to either grow or shrink
                            % toward 0, but not flip. Do this by limiting
                            % gaze_bias*temporary_bias_toward_S to [-1,1].
                    weights_modulated_by_gaze(greed_index) = ...
                        weights_modulated_by_gaze(greed_index) + gaze_bias * (relative_gaze(2) - .5);
                end
                if ~isempty(risk_index)
%                     weights_modulated_by_gaze(risk_index) = ...
%                         weights_modulated_by_gaze(risk_index) * (1 + max(min(gaze_bias*temporary_bias_toward_S,1),-1));
%                     weights_modulated_by_gaze(risk_index) = ...
%                         weights_modulated_by_gaze(risk_index) * (1 + gaze_bias*temporary_bias_toward_S);
                            % We want this weight to either grow or shrink
                            % toward 0, but not flip. Do this by limiting
                            % gaze_bias*temporary_bias_toward_S to [-1,1].
                    weights_modulated_by_gaze(risk_index) = ...
                        weights_modulated_by_gaze(risk_index) + gaze_bias * (relative_gaze(1) - .5);
                end
                weights_modulated_by_gaze_updating(ti,:) = weights_modulated_by_gaze;
%                 coop_value = sum(weights_modulated_by_gaze.*basis_function_choices);
%                 weights = weights_modulated_by_gaze;                
                coop_value_gaze = sum(weights_modulated_by_gaze.*basis_function_choices);
                    % gaze-adjusted value for current choice
                    
                % Apply softmax decision rule for CURRENT CHOICE (adjusted by gaze)
                model_prediction_distribution = softmax([coop_value_gaze, 0], inv_temp);
                    % So critically here the gaze only adjusts temporarily the
                    % choice likelihoods, without influencing the value
                    % estimates or the weights that are brought to the next
                    % trial
            else
                % Apply softmax decision rule for CURRENT CHOICE
                model_prediction_distribution = softmax([coop_value, 0], inv_temp);
            end
               
            model_coop_likelihood_grid(Ss == S, Ts == T) = model_prediction_distribution(1);
            store_pred_dist((bi-1)*16+ti,:) = model_prediction_distribution;
            
            % Evaluate probability distribution to sample a choice
            model_prediction = model_prediction_distribution(1)>rand;
            store_model_prediction((bi-1)*16+ti) = model_prediction;
            
            %%% LEARNING
            
            % Outcome
            outcome = strcmp(block_dat.CorrAns{ti},'coop'); % Write as 0, 1
            player_choice_grid(Ss == S, Ts == T) = 2 * outcome - 1; % Write as -1, 1
            
            % Prediction error
                % The value estimate that goes into P.E. is NEVER adjusted by gaze
                % So let's compute it again:
            model_prediction_distribution = softmax([coop_value, 0], inv_temp);
            PE = outcome - model_prediction_distribution(1);
            
            % Update
            weights_updates = basis_function_choices * PE;
            weights = weights + weights_updates .* LRs(1 + double(weights_updates > 0))';
            if bounded_weights
                weights = max(weights,0);
            end
            weights_updating(ti+1,:) = weights;

            %%% VIZ
            if visualize
                figure(fig);
                
                subplot(2,6,7:8);
                imagesc('CData',player_choice_grid);
                caxis([-1,1]);
                xlabel('T'); ylabel('S');
                xticks(1:11); yticks(1:11);
                xticklabels(Ts); yticklabels(Ss);
                title(sprintf('Player choices %s',player_type));
                axis equal;
                axis tight;
                colorbar;

                subplot(2,6,1:3);
                bar(weights);
                xticklabels(basis_sub_set_names);
                xtickangle(45);
                ylabel('Weight'); title('Basis set weights');

                subplot(2,6,9:10);
                imagesc('CData',participant_prediction_confidence_joint_grid);
                caxis([0,1]);
                xlabel('T'); ylabel('S');
                xticks(1:11); yticks(1:11);
                xticklabels(Ts); yticklabels(Ss);
                title(sprintf('Subject %s''s predictions (1 = C, 0 = D)',sub_ID));
                axis equal;
                axis tight;
                colorbar;
                
                subplot(2,6,11:12);
                hold on;
                imagesc('CData',model_coop_likelihood_grid)
                caxis([0,1]);
                xlabel('T'); ylabel('S');
                xticks(1:11); yticks(1:11);
                xticklabels(Ts); yticklabels(Ss);
                title('Model predictions');
                axis equal;
                axis tight;
                colorbar;
                hold on;
                for ti_tmp = 1:16
                    S_tmp = block_dat.S(ti_tmp);
                    T_tmp = block_dat.T(ti_tmp);
                    text(find(Ts == T_tmp), find(Ss == S_tmp), sprintf('%i',ti_tmp),...
                        'color','white','HorizontalAlignment','center');
                end

                if ti == 16
                    pause(.05);
                else
                    pause(.05);
                end
                
                u = uicontrol('Style','slider','Position',[10 50 20 340],...
                    'Min',1,'Max',64,'Value',1);
                u.Value = (bi-1)*16+ti;
                M((bi-1)*16+ti) = getframe(gcf);
                
            end
        end
        if visualize
            pause(1); % End-of-block pause
        end
        model_coop_likelihoods_per_block.(sprintf('block_%i',bi)) = model_coop_likelihood_grid;
        participant_prediction_confidence_joint_grid_per_block.(sprintf('block_%i',bi)) = participant_prediction_confidence_joint_grid;
        weights_updating_per_block.(sprintf('block_%i',bi)) = weights_updating;
        if gaze
            weights_modulated_by_gaze_per_block.(sprintf('block_%i',bi)) = weights_modulated_by_gaze_updating;
        end
    end

    %% Compute likelihood of data given model

    % Choice
    columns = 2 - strcmp(sub_dat.GivenAns,'coop'); % Column 1 for coop, col 2 for def
    rows = (1:length(store_pred_dist))';
    indices = sub2ind(size(store_pred_dist),rows,columns);
    choice_probabilities_model = store_pred_dist(indices);
    choice_NLL = -sum(log(choice_probabilities_model));
    if choice_NLL == Inf
        choice_NLL = -log(realmin);
    end
    
    if strcmp(fit_to, 'joint')
        
        
        model_errors = store_participant_prediction_confidence_joint - store_pred_dist(:,1);
        SSE = sum(model_errors.^2);
        
        cost = SSE;
        cost_type = 'SSE';
        joint_NLL = -sum(log(normpdf(model_errors, 0, std(model_errors))));
        n_obs = height(sub_dat);
        BIC = -2 * -joint_NLL + n_params * log(n_obs);
        AIC = -2 * -joint_NLL + 2*n_params;
                
        % Randomly sample predictions
        sampled_predictions = (store_pred_dist(:,1) + random('norm',0,std(model_errors),n_obs,1)) >= 0.5;
        
    elseif strcmp(fit_to,'choice_only')
        cost = choice_NLL;
        cost_type = 'NLL';
        n_obs = height(sub_dat);
        BIC = -2 * -choice_NLL + n_params * log(n_obs);
        AIC = -2 * -choice_NLL + 2*n_params;
        
        % Randomly sample predictions
        sampled_predictions = [];
    end
    
    if returnInfo
        
        % Store variables
        input_args.params = params;
        input_args.sub_ID = sub_ID;
        input_args.sub_dat = sub_dat;
%         input_args.trial_gaze_diffs = trial_gaze_diffs;
        input_args.relative_gazes = relative_gazes;
        input_args.basis_sub_set = basis_sub_set;
        input_args.basis_sub_set_names = basis_sub_set_names;
        input_args.bounded_weights = bounded_weights;
        input_args.asymmetric_LR = asymmetric_LR;
        input_args.fit_to = fit_to;
        input_args.gaze = gaze;
        input_args.visualize = visualize;
        input_args.returnInfo = returnInfo;
        out.input = input_args;
        out.cost = cost;
        out.cost_type = cost_type;
        out.choice_NLL = choice_NLL;
        out.fit_to = fit_to;
        if strcmp(fit_to,'joint')
            out.SSE = SSE;
            out.NLL = joint_NLL;
        elseif strcmp(fit_to,'choice_only')
            out.NLL = choice_NLL;
        end
        out.BIC = BIC;
        out.AIC = AIC;
        out.choice_probabilities_model = choice_probabilities_model;
        out.store_coop_values = store_coop_values;
        out.store_pred_dist = store_pred_dist;
        out.store_model_prediction = store_model_prediction;
        out.store_sampled_model_prediction = sampled_predictions;
        out.store_participant_prediction_confidence_joint = store_participant_prediction_confidence_joint;
        out.participant_prediction_confidence_joint_grid_per_block = participant_prediction_confidence_joint_grid_per_block;
        out.model_coop_likelihoods_per_block = model_coop_likelihoods_per_block;
        out.weights_updating_per_block = weights_updating_per_block;
        out.basis_sub_set = basis_sub_set;
        if visualize
            out.figure_frames = M;
        end
        if gaze
            out.info_values = info_values;
            out.weights_modulated_by_gaze_per_block = weights_modulated_by_gaze_per_block;
        end
    else
        out = cost;
    end

end