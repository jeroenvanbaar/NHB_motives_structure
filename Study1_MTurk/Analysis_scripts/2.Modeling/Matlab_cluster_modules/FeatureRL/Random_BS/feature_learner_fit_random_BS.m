%% Fit feature learner with random basis set
% Fits to choice and confidence data using joey style joint metric

function feature_learner_fit_random_BS(subInd, n_random_basis_sets, varargin)
    
    %% Fixed arguments
    fitTo = 'joint';
    asymmetric_LR = false;
    bounded_weights = false;
    n_random_bases_per_set = 4;
    
    %% Parse variable arguments
    basedOnTrueBases = false;
    for varg = 1:length(varargin)
        if ischar(varargin{varg})
            if strcmpi('basedOnTrueBases',varargin{varg})
                basedOnTrueBases = varargin{varg + 1};
                varargin{varg} = {}; varargin{varg + 1} = {};
            end
        end
    end
    
    %% Game variables; softmax choice rule
    global Ss Ts softmax % entropicConfidence
    Ss = [0,3,7,10];
    Ts = [5,8,12,15];
    softmax = @(values, invTemp) exp(values.*invTemp)./sum(exp(values.*invTemp));

    %% Load data for subject

    opts = detectImportOptions('../../gameDat_total.csv');
    opts = setvartype(opts,{'Block','Trial','S','T',...
        'ConfidenceNum','ScoreNum'},'single');
    opts = setvartype(opts,{'subID','Player', 'Type','Variant',...
        'Type_Total','GameType','Colors','GivenAns','CorrAns',...
        'SelfReport'},'string');
    clear gameDat
    gameDat = readtable('../../gameDat_total.csv',opts,'ReadRowNames',true);
    gameDat.Properties.VariableNames{16} = 'Confidence';
    gameDat.Properties.VariableNames{17} = 'Score';
    gameDat{gameDat.Trial<8,'Round'} = 1;
    gameDat{gameDat.Trial<4,'Round'} = 0;
    gameDat{gameDat.Trial>7,'Round'} = 2;
    gameDat{gameDat.Trial>11,'Round'} = 3;
    gameDat{:,'Phase'} = 0; % == 'Early'
    gameDat{gameDat.Round>1,'Phase'} = 1; % == 'Late'
    subIDs = unique(gameDat.subID);
    subID = subIDs(subInd);
    fprintf('Sub ID: %s\n',subID);
    subDat = gameDat(gameDat.subID==subID,:);
    
    %% Open log / results files
    results_dir = sprintf('Results/results_%s',datestr(now,'yyyy-mm-dd'));
    fprintf('Using results directory %s\n',results_dir);
    if ~exist(results_dir,'dir')
        mkdir(results_dir);
        fprintf('Created results directory %s\n',results_dir);
    end
    log_filename_base = sprintf('%s/subInd-%i_subID-%s_nBases-%i_nSets-%i_log.csv',...
        results_dir, subInd, subID, n_random_bases_per_set, n_random_basis_sets);
    filename = log_filename_base;
    suffix = 1;
    while exist(filename,'file') == 2
        suffix = suffix + 1;
        filename = [log_filename_base(1:end-4) sprintf('_%i.csv',suffix)];
    end
    logID = fopen(log_filename_base,'a');
    header_line = ['subInd,subID,fitTo,asymm_LR,bounded_weights,', ...
        sprintf('basis_%i_hex,',1:n_random_bases_per_set), ...
        'LR,invTemp,' sprintf('feature_weight_%i,',1:n_random_bases_per_set) 'SSE\n'];
    fprintf(logID, header_line);

    %% Model settings
    
    returnInfo = false;
    clear options
    options = optimoptions('fmincon');
    options.Display = 'iter';
    options.FunctionTolerance = 0.1;
    options.OptimalityTolerance = 0.01;
    A = []; b = []; Aeq = []; beq = [];
    
    fprintf(['Fitting featureRL model to %s, asymmetric LR %s, ',...
            'bounded weights %s, %i random bases. %s\n'], ...
            fitTo, string(asymmetric_LR), ...
            string(bounded_weights), n_random_bases_per_set, datetime);

    % Parameter bounds
    paramBounds = [repmat([0.01   , 10],(1 + double(asymmetric_LR)),1); ... % LR
                   0.01   , 5; ... % InvTemp
                   repmat([0 - (1 - double(bounded_weights)) * 20,20],...
                        n_random_bases_per_set,1); ... % Weights
                   ];
    lb = paramBounds(:,1);
    ub = paramBounds(:,2);
    
    %% Loop over basis sets
    
    % Prepare output storage
    out = cell(n_random_basis_sets, 11);
    if basedOnTrueBases
        load(sprintf('random_bases_%i_shuffleTrueBasisSet.mat',n_random_basis_sets),'random_bases');
    else
        load(sprintf('random_bases_%i.mat',n_random_basis_sets),'random_bases');
    end
    
    for random_basis_set_i = 1:n_random_basis_sets
        
        % Load random basis set
        random_basis_set = zeros(4,4,n_random_bases_per_set);
        for random_basis_i = 1:n_random_bases_per_set
            random_basis_set(:,:,random_basis_i) = reshape(random_bases(random_basis_set_i,random_basis_i,:,:),4,4);
        end

        % Store as set and hex
        basisSet = struct('name',[],'grid',[]);
        basisSet_hex = cell(1,n_random_bases_per_set);
        for random_basis_i = 1:n_random_bases_per_set
            basisSet(length(basisSet)+1).name = sprintf('random_basis_%i',random_basis_i);
            random_basis = random_basis_set(:,:,random_basis_i);
            basisSet(length(basisSet)).grid = random_basis;
            basisSet_hex{random_basis_i} = [strcat(dec2hex(find(random_basis==1)-1)') '_' ...
                                            strcat(dec2hex(find(random_basis==.5)-1)')];
        end
        basisSet = basisSet(2:end); % Clear out empty first field

        % Fit
        objFunc = @(params)SSE_featureLearner_random_BS(params, subDat, basisSet, returnInfo);
        params0 = lb + rand(size(paramBounds,1),1).*(ub-lb);
        [x,SSE] = fmincon(objFunc, params0, A, b, Aeq, beq, lb, ub, [], options);

        % Store iteration
        % % For .mat
        out{random_basis_set_i,1} = subInd;
        out{random_basis_set_i,2} = subID;
        out{random_basis_set_i,3} = fitTo;
        out{random_basis_set_i,4} = 0;
        out{random_basis_set_i,5} = 0;
        out(random_basis_set_i,6:(5+n_random_bases_per_set)) = basisSet_hex;
        out{random_basis_set_i,5 + n_random_bases_per_set + 1} = x;
        out{random_basis_set_i,5 + n_random_bases_per_set + 2} = SSE;
        % % For .csv
        log_line = [sprintf('%i,%s,%s,%i,%i',subInd,subID,fitTo,0,0) ...
                    sprintf(',%s',basisSet_hex{:}) ...
                    sprintf(',%.3f',x) sprintf(',%.3f',SSE) '\n' ...
                    ];
        fprintf(logID, log_line);
        % % Display
        disp([SSE, x']);
    
    end
    
    % To plot:
%     best_index = 18;
%     Plot_bases_from_hex(out(best_index,6:9),out{best_index,10}(3:end));
    
    % Store all results as .mat
    results_filename_base = sprintf('%s/subInd-%i_subID-%s_nBases-%i_nSets-%i_results.mat',...
        results_dir, subInd, subID, n_random_bases_per_set, n_random_basis_sets);
    filename = results_filename_base;
    suffix = 1;
    while exist(filename,'file') == 2
        suffix = suffix + 1;
        filename = [results_filename_base(1:end-4) sprintf('_%i.mat',suffix)];
    end
    save(filename,'out');

end


