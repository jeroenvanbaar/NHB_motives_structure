%% Collect SSEs

all_out_all = {};
for sub_index = 201:353
    sub_ID = sub_IDs(sub_index);
    fprintf('Sub ID: %s\n',sub_ID);
    sub_dat = game_dat(game_dat.subID==sub_ID,:);
    fit_to = 'joint_joey_style';

    niter = 5;
    model_results = load(sprintf('Results/results_03-Oct-2019/subInd-%i_subID-%s_niter-%i_results.mat',sub_index,sub_ID,niter));
    out_all = model_results.out_all;
    % NLL
    NLLs = cell2mat(out_all(:,7));
    model_selectors.NLL = NLLs;
    % BIC_NLL
    nParams = sum(~isnan(cell2mat(out_all(:,8))),2);
    BICs = penalizedmodelfit(-NLLs, 128, nParams, 'metric','BIC','type','LLE');
    model_selectors.BIC_NLL = BICs;
    % SSE
    SSEs = nan(length(out_all),1);
    for mi = 1:length(out_all)
        model = out_all(mi,:);
        combInd = model{3};
        asymmetric_LR = model{5}; bounded_weights = model{6};
        params = model{8}';
        comb = combs(combInd,:);
        basisSubSet = basisSet(comb=='1');
        % Run model again
        gaze = false; visualize = false; returnInfo = true;
        out = NLL_featureLearner_3(params, ... % Free parameters
            sub_ID, sub_dat, ... % Input data
            basisSubSet, bounded_weights, asymmetric_LR, fit_to, ... % Model structure
            gaze, visualize, returnInfo);
        SSEs(mi) = out.SSE;
    end
    out_all(:,9) = num2cell(SSEs);
    all_out_all = [all_out_all;[cellstr(repmat(sub_ID,length(out_all),1)),out_all]];
    model_selectors.SSE = SSEs;
    %BIC based on SSE
    BICs_SSE = penalizedmodelfit(SSEs,128,nParams,'type','SSE','metric','BIC');
    model_selectors.BIC_SSE = BICs_SSE;
end

save('Results/results_03-Oct-2019/all_results_incl_SSE.mat','all_out_all');

%% Newer models

opts = detectImportOptions('../gameDat_total.csv');
opts = setvartype(opts,{'Block','Trial','S','T',...
    'ConfidenceNum','ScoreNum'},'single');
opts = setvartype(opts,{'subID','Player', 'Type','Variant',...
    'Type_Total','GameType','Colors','GivenAns','CorrAns',...
    'SelfReport'},'string');
clear gameDat
game_dat = readtable('../gameDat_total.csv',opts,'ReadRowNames',true);
game_dat.Properties.VariableNames{16} = 'Confidence';
game_dat.Properties.VariableNames{17} = 'Score';
game_dat{game_dat.Trial<8,'Round'} = 1;
game_dat{game_dat.Trial<4,'Round'} = 0;
game_dat{game_dat.Trial>7,'Round'} = 2;
game_dat{game_dat.Trial>11,'Round'} = 3;
game_dat{:,'Phase'} = 0; % == 'Early'
game_dat{game_dat.Round>1,'Phase'} = 1; % == 'Late'
sub_IDs = unique(game_dat.subID);

date_string = '2019-10-21';
%%
all_out = {};
for sub_index = 1:200
    if mod(sub_index,10) == 0
        disp(sub_index);
    end
    sub_ID = sub_IDs(sub_index);
    fprintf('Sub ID: %s\n',sub_ID);
    sub_dat = game_dat(game_dat.subID==sub_ID,:);
    fit_to = 'joint';

    niter = 5;
    load(sprintf('Results/results_%s/subInd-%i_subID-%s_niter-%i_results.mat',...
        date_string,sub_index,sub_ID,niter),'out');
    
    all_out = [all_out;out];
end

header_line = ['sub_ind,sub_ID,fit_to,comb_index,comb,feature_names,asymm_LR,bounded_weights,', ...
        'LR_up,LR_down,inv_temp,' sprintf('feature_weight_%i,',1:4) 'cost_type,cost'];
colnames = strsplit(header_line,',');
data_table_root = cell2table(all_out(:,1:11),'VariableNames',colnames(1:11));
data_table_ap1 = cell2table(num2cell(cell2mat(all_out(:,12))),'VariableNames',colnames(12:15));
data_table_ap2 = cell2table(all_out(:,13:end),'VariableNames',colnames(16:end));
data_table = horzcat(data_table_root,data_table_ap1,data_table_ap2);

writetable(data_table,sprintf('Results/results_%s/all_results.csv',date_string));






