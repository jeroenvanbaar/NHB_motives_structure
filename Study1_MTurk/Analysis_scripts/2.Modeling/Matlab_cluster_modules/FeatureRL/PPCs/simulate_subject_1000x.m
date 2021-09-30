%% POSTERIOR PREDICTIVE CHECKS FOR MODEL
%
% -------------------------------------------------------------------------
% Goals:
% -------------------------------------------------------------------------
%
% 1. Simulate model 1000x for each subject
% 2. Compare distribution of mean simulated performance to true performance
% 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SETUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load functions
addpath('..','../helpers');
cd('..');
% Game variables; softmax choice rule
define_globals;
% Load data
[game_dat, sub_IDs] = load_game_dat;
% Load model results
study_name = 'Study1_Mturk';
date_string = '2020-03-24';
features = 'CoGrRiNa';
model_string = 'FeatureRL';
fit_to = 'joint';
niter = 10;
gaze = 'False';
filename = ['/Users/jeroen/Dropbox (Brown)/Postdoc FHL/JEROEN/SOC_STRUCT_LEARN/',...
         'Study1_MTurk/Data/Cleaned/Model_results/',...
         sprintf('%s_%s_%s_fitto-%s_gaze-%s_niter-%i.csv',...
             model_string,features,date_string, fit_to, gaze, niter)];
opts = detectImportOptions(filename);
opts = setvartype(opts,{'sub_ind'},'single');
opts = setvartype(opts,{'sub_ID','comb','fit_to'},'string');
clear model_fits
model_results = readtable(filename,opts,'ReadRowNames',false);

% Preallocate
out_all = {};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Run simulations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subs = 1:150;
niter = 1000;
results = nan(length(subs)*niter,6);

for subi = 1:length(subs)
    sub_index = subs(subi);
    disp(sub_index);
    sub_ID = sub_IDs(sub_index);
    sub_dat = game_dat(game_dat.subID==sub_ID,:);
    sub_model = model_results(model_results.sub_ID == sub_ID,:);
    best_model = sub_model(sub_model.BIC == min(sub_model.BIC),:);
    comb = best_model.comb;

    % Get features
    features_string = best_model.feature_names{1};
    features_split = strsplit(features_string,'_');
    n_features = length(features_split);
    features = '';
    for fi = 1:n_features
        features = [features, features_split{fi}(1:2)];
    end

    % Build basis set for model
    [basis_sub_set,~] = define_basis_set_2(features);
    basis_sub_set_names = strip(sprintf('%s_',basis_sub_set.name),'_');

    % Get params
    params = table2array(best_model(:,[10,12,14,15,16,17]))';

    % Run model
    visualize = false;
    gaze = false; return_info = true;
    for i = 1:niter
        out = cost_function_featureLearner_6(params, ... % Free parameters
            sub_ID, sub_dat, [], ... % Input data
            basis_sub_set, 0, 0, 'joint', ... % Model structure
            gaze, visualize, return_info);
%         model_pred = out.store_model_prediction;
        model_pred = out.store_sampled_model_prediction;
        true_acc_nat = sum(sub_dat.CorrAns(1:32) == sub_dat.GivenAns(1:32))/32;
        sim_acc_nat = sum(model_pred(1:32) == (sub_dat.CorrAns(1:32) == "coop"))/32;
        true_acc_inv = sum(sub_dat.CorrAns(33:64) == sub_dat.GivenAns(33:64))/32;
        sim_acc_inv = sum(model_pred(33:64) == (sub_dat.CorrAns(33:64) == "coop"))/32;
        results((sub_index-1)*niter + i,:) = [sub_index,i,true_acc_nat,sim_acc_nat,true_acc_inv,sim_acc_inv];
    end
    disp(mean(results(((sub_index-1)*niter+1):(sub_index*niter),3:end)));
end
% csvwrite('~/Desktop/sim_performance.csv',results);

%%
mean_results = nan(length(subs),5);
for subi = 1:length(subs)
    sub_index = subs(subi);
    mean_results(sub_index,:) = [sub_index,mean(results(((sub_index-1)*niter+1):(sub_index*niter),3:end))];
end

figure;
scatter(rand(length(subs),1),mean_results(:,2));
hold on
scatter(2+rand(length(subs),1),mean_results(:,3));
scatter(5+rand(length(subs),1),mean_results(:,4));
scatter(7+rand(length(subs),1),mean_results(:,5));
