
function Step2_fit_simulated_data(smi, sim_index, niter, plot_simulation_choice)

    %% Prep
%     addpath('..');
    addpath('../FeatureRL/helpers');
    define_globals;

    % Select simulation data
    load('simulations.mat');
    load('params.mat');
    
    % Select model
    model_feature_sets = {'CoST','CoHgShSgPd','CoGrRiNa','motives','players'};
    simulated_model_feature_set = model_feature_sets{smi};
    sim_dat = all_simulations.(simulated_model_feature_set).(sprintf('sub_%i',sim_index));
    params = all_params.(simulated_model_feature_set).(sprintf('sub_%i',sim_index));
    
    %% Define model function
    if ~sum(strcmp(simulated_model_feature_set,{'motives','players'})) && plot_simulation_choice
        plot_simulation(sim_dat);
    end
    
%     %% Open results file
%     fid = fopen(sprintf('results/results_smi-%i_sim_index-%i.txt',smi,sim_index),'w');
%     header = sprintf('smi,sim_index,mfsi,model_feature_set,params_original,params_recovered,SSE,BIC\n');
%     fwrite(fid,header);
    
    %% Fit models    
    for mfsi = 1:length(model_feature_sets)
        
        model_feature_set = model_feature_sets{mfsi};
        disp(model_feature_set);
        
        % Find model type
        if sum(strcmp(model_feature_set,{'motives','players'}))
            addpath(genpath('../BayesianIO/'));
            model_function = @NLL_SocPredIdealObs;
            
            % Find model details
            switch model_feature_set
                case 'players'
                    stateNames  = {'Opt','Pess','InvOpt','InvPess'};
                case 'motives'
                    stateNames  = {'Trust','Opt','Pess','ExpVal'};
                case 'poncela'
                    stateNames  = {'Trust','Opt','Pess','Env'};
            end

            % Parameter bounds
            paramBounds = repmat([0,1],length(stateNames),1);
            lb = paramBounds(:,1);
            ub = paramBounds(:,2);
            A = [1,1,1,0]; b = 0.999; Aeq = []; beq = [];
            
            % Determine objective function
            objective_function = @(params)NLL_SocPredIdealObs(params, [], sim_dat, ...
                stateNames, 'joint', 0, 0);
            % Fit
            options = optimoptions('fmincon');
%             options.Display = 'iter';
            options.DiffMinChange = .001;
            costs = nan(niter,1);
            xs = nan(niter,length(lb));
            n = 1;
            while n <= niter
                try
                    params0 = nan(size(paramBounds,1),1);
                    initPriors = rand(4,1);
                    initPriors = initPriors ./ sum(initPriors);
                    params0(1:3) = initPriors(1:3); % This ensures that the sum of the priors does not exceed 1, allowing the fourth prior to be computed
                    params0(4) = lb(4) + rand(1,1).*ub(4);
                    [x,cost] = fmincon(objective_function, params0, ...
                        A, b, Aeq, beq, lb, ub, [], options);
                    costs(n) = cost;
                    xs(n,:) = x';
                    n = n + 1;
                catch
                    disp('Error - skipping');
                end
            end
            [~,best_iter] = min(costs);
            x = xs(best_iter,:)';
            % Get BIC
            model_out = model_function(x, [], sim_dat, ...
                stateNames, 'joint', 1, 0);
            if strcmp(model_out.costType,'SSE')
                SSE = model_out.cost;
            else
                error('No SSE computed');
            end
            LL = -model_out.totalNLL;
            BIC = model_out.BIC;
            AIC = model_out.AIC;
        else
            addpath(genpath('../FeatureRL/'));
            model_function = @cost_function_featureLearner_6;
            
            % Find model details
            [basis_set,~] = define_basis_set_2(model_feature_set);
%             n_features_full_basis_set = length(basis_set);
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
            
            % Fit
            objective_function = @(params) model_function(params, ... % Free parameters
                [], sim_dat, [], ... % Input data
                basis_set, 0, 0, 'joint', ... % Model structure
                0, 0, false);
            options = optimoptions('fmincon');
%             options.Display = 'iter';
            costs = nan(niter,1);
            xs = nan(niter,length(lb));
            for n = 1:niter
                params0 = lb + rand(size(param_bounds,1),1).*(ub-lb);
                [x,cost] = fmincon(objective_function, params0, ...
                    [], [], [], [], lb, ub, [], options);
                costs(n,:) = cost;
                xs(n,:) = x';
            end
            [~,best_iter] = min(costs);
            x = xs(best_iter,:)';
            % Get BIC
            model_out = model_function(x, ... % Free parameters
                [], sim_dat, [], ... % Input data
                basis_set, 0, 0, 'joint', ... % Model structure
                0, 0, true);
            SSE = model_out.SSE;
            LL = -model_out.NLL;
            BIC = model_out.BIC;
            AIC = model_out.AIC;
        end
        
%         if plot_result
%             figure;
%             subplot(2,2,1);
%             plot(params);
%             subplot(2,2,3);
%             plot(x);
%             subplot(2,2,[2,4]);
%             scatter(params,x);
%         end

        % Store
%         writeline = [sprintf('%i,%i,%i,%s,',smi,sim_index,mfsi,model_feature_set),...
%             sprintf('%.2f',params,x,SSE,BIC), '\n'];
%         fwrite(fid,writeline);
        results_mat = struct();
        results_mat.smi = smi; % simulated model index
        results_mat.sim_index = sim_index;
        results_mat.mfsi = mfsi; % feature set index
        results_mat.model_feature_set = model_feature_set;
        results_mat.original_params = params; 
        results_mat.recovered_params = x;
        results_mat.SSE = SSE;
        results_mat.LL = LL;
        results_mat.BIC = BIC;
        results_mat.AIC = AIC;
        save(sprintf('results/results_smi-%i_sim-%i_mfsi-%i.mat',smi,sim_index,mfsi),'results_mat');
        fprintf('Results written including AIC.\n');
        
    end
end
