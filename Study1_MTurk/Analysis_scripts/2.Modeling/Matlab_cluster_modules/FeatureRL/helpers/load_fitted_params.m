%% Load fitted parameters

function out = load_fitted_params(model_type,datestr,sub_ID,model_feature_set)
    param_dir = sprintf('/gpfs/home/jvanbaar/data/jvanbaar/SOC_STRUCT_LEARN/ComputationalModel/%s/Results/results_%s',...
        model_type,datestr);

    switch model_type
        case 'FeatureRL'
            a = ls([param_dir, sprintf('/*%s*%s*.csv',sub_ID,model_feature_set)]);
            strvars = {'sub_ID','fit_to', 'comb','feature_names','cost_type'};
        case 'BayesianIO'
            a = ls([param_dir, sprintf('/params*%s*.csv',sub_ID)]);
            strvars = {'subID','fitTo', 'classSet','costType'};
    end
    
    opts = detectImportOptions(a(1:end-1));
    opts = setvartype(opts,'single');
    opts = setvartype(opts,strvars,'string');
    clear params; params = readtable(a(1:end-1),opts,'ReadRowNames',false);
    
    clear out;
    out.subID = sub_ID;
    out.model_type = model_type;
    out.datestr = datestr;
    out.features = model_feature_set;
    switch model_type
        case 'FeatureRL'            
            out.LR = params{end,'LR_up'};
            out.inv_temp = params{end,'inv_temp'};
            out.feature_weights = params{end,contains(params.Properties.VariableNames, 'feature_weight_')};
        case 'BayesianIO'
            try
                out.noise = params{strcmp(params.classSet,model_feature_set),'prior4'};
            catch
                out.noise = params{strcmp(params.classSet,model_feature_set),'noise'};
            end
            out.priors = params{strcmp(params.classSet,model_feature_set),{'prior1','prior2','prior3'}};
    end
end