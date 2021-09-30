%% Plot results random basis sets

subInd = 1;
n_sets = 5000;
filename = dir(sprintf('Results/results*/subInd-%i_*nSets-%i_results.mat',subInd,n_sets));
out = load([filename.folder, '/', filename.name]);
out = out.out;
[best_SSE,best_index] = min(cell2mat(out(:,11)));
weighted_mean = Plot_bases_from_hex(out(best_index,6:9),out{best_index,10}(3:end),1,1);

%% All subjects
subInds = 1:150;
n_subs = length(subInds);
all_best_models = cell(n_subs,11);
all_weighted_means = nan(4,4,n_subs);

%% Load best for all subjects
for subInd = subInds
    if mod(subInd,10) == 0
        disp(subInd);
    end
    filename = dir(sprintf('Results/results*18/subInd-%i_*.mat',subInd));
    out = load([filename.folder, '/', filename.name]);
    out = out.out;
    [best_SSE,best_index] = min(cell2mat(out(:,11)));
    best_model = out(best_index,:);
    all_best_models(subInd,:) = best_model;
    if subInd < 1
        weighted_mean = Plot_bases_from_hex(best_model(6:9),best_model{10}(3:end), true);
        title(sprintf('Sub index %i',subInd));
    else
        weighted_mean = Plot_bases_from_hex(best_model(6:9),best_model{10}(3:end), false);
    end
    all_weighted_means(:,:,subInd) = weighted_mean;
end

%% Load all results
load('random_bases_5000_shuffleTrueBasisSet.mat','random_bases');
SSEs_all = nan(5000,n_subs);
BICS_all = nan(5000,n_subs);
weights_all = nan(5000,n_subs,4);
for subInd = subInds
    if mod(subInd,10) == 0
        disp(subInd);
    end
    filename = dir(sprintf('Results/results*18/subInd-%i_*.mat',subInd));
    out = load([filename.folder, '/', filename.name]);
    out = out.out;
    SSEs = cell2mat(out(:,11));
    SSEs_all(:,subInd) = SSEs;
    weights = reshape(cell2mat(out(:,10)),6,5000)';
    weights_all(:,subInd,:) = weights(:,3:end);
end

%% Plot all randomized results
mean_SSEs = mean(SSEs_all,2);
[best_SSE,best_iter] = min(mean_SSEs);

hist_fig = figure;
figure(hist_fig)
hist_dat = histogram(mean_SSEs,200);
title('Model fits for random models');
xlabel('Mean SSE across subjects');
ylabel('Frequency');

%% Add in true model results

load('/gpfs/home/jvanbaar/data/jvanbaar/SOC_STRUCT_LEARN/ComputationalModel/FeatureRL/Results/results_03-Oct-2019/all_results_incl_SSE.mat',...
    'all_out_all');
right_comb = strcmp(all_out_all(:,5),'1111');
right_class = (cell2mat(all_out_all(:,6)) == 0) & (cell2mat(all_out_all(:,7)) == 0); % Asymmetric LR, bounded weights -- both set to 0 for the final model
models_to_compare = all_out_all(right_comb & right_class,:);
mean_SSE_psych_model = mean(cell2mat(models_to_compare(1:150,10)));

fname = ['/gpfs/home/jvanbaar/data/jvanbaar/SOC_STRUCT_LEARN/ComputationalModel/FeatureRL/Results/final_model_results_for_paper/Study1_Mturk/',...
    'FeatureRL_CoGrRiNa_2020-03-24_fitto-joint_gaze-false_niter-10.csv'];
opts = detectImportOptions(fname);
opts = setvartype(opts, {'comb'},{'string'});
all_out_all = readtable(fname,opts);
right_comb = strcmp(all_out_all{:,5},'1111');
right_class = (all_out_all{:,7} == 0) & (all_out_all{:,8} == 0); % Asymmetric LR, bounded weights -- both set to 0 for the final model
models_to_compare = all_out_all(right_comb & right_class,:);

mean_SSE_psych_model = mean(models_to_compare.SSE);

figure(hist_fig);
hold on;
best_4_features_line = plot([mean_SSE_psych_model, mean_SSE_psych_model], ylim, 'r--', 'linewidth', 2);
pvalue_SSE_psych_model = 1 - sum(mean_SSE_psych_model < mean_SSEs)/length(mean_SSEs);
legend({'5000 random basis sets', 'Canonical motives'});

print(hist_fig,'random_bs_results.png','-dpng','-r300');

%% Add in true model results from CoGrRiNa basis set
mean_SSE_newbases = 4.2569; % Load this from model fit on 2019-10-28 with features CoGrRiReNaEn

hist_fig = figure;
figure(hist_fig)
hist_dat = histogram(mean_SSEs,200, 'EdgeColor','none', 'FaceColor',[.4,.4,.4]);
xlabel(''); xticks(0:5:25);
xlim([0,30]);

hold on;
best_4_features_line = plot([mean_SSE_newbases, mean_SSE_newbases], ylim, 'k', 'linewidth', 2);
pvalue_SSE_psych_model = 1 - sum(mean_SSE_newbases < mean_SSEs)/length(mean_SSEs);
% legend({'Random motives', '4 canonical motives'});
ylabel(''); yticks([]);
% export_fig 'random_bs_results_vs_CoGrRiNa.eps' %-transparent 
print(hist_fig,'random_bs_results_vs_CoGrRiNa.png','-dpng','-r300');

%% Select best random basis set

[sorted,sorti] = sort(mean_SSEs);

for best_i = 1:3

    best_basis_set = out(sorti(best_i),6:9);
    best_random_fig = figure();
    mean_weights_on_best_basis_set = mean(reshape(weights_all(sorti(best_i),:,:),150,4),1);
    weighted_mean = Plot_bases_from_hex(best_basis_set,mean_weights_on_best_basis_set, true, best_random_fig);
    if best_i == 1
        print(best_random_fig,'best_random_basis_set.eps','-depsc');
    end
end

%% Plot a few middle-performing basis sets

figure;
for best_i = 1:10

    best_basis_set = out(sorti(2500-1+best_i),6:9);
    best_random_fig = figure();
    mean_weights_on_best_basis_set = mean(reshape(weights_all(sorti(2500-1+best_i),:,:),150,4),1);
    weighted_mean = Plot_bases_from_hex(best_basis_set,mean_weights_on_best_basis_set, true, best_random_fig);
%     print(best_random_fig,'middlet_random_basis_set.png','-dpng','-r300');

end

%% Store
save('Results/all_best_models_nBases-4_nSets-5000_shuffled.mat','all_best_models');
% save('Results/all_best_models_nBases-4_nSets-1000.mat','all_best_models');

%% Load
load('Results/all_best_models_nBases-4_nSets-1000.mat','all_best_models');


%% Try to integrate meaningfully

% Overlay all 'most successful' bases?
grids = nan(4,4,n_subs*4);
hexes = all_best_models(:,6:9);
hexes = hexes(:);
weights = [];
for subInd = subInds
    weights = [weights; all_best_models{subInd,10}(3:end)];
end
% n_bases = length(basisSet_hex);
% grids = nan(4,4,n_bases);
for i = 1:length(grids)
    hex = hexes{i};
    grid = zeros(4,4);
    for pos = 1:8
        index = hex2dec(hex(pos)) + 1;
        grid(index) = 1;
    end
    grid = grid.*weights(i);
    grid = grid + min(min(grid));
    grids(:,:,i) = grid;
end

grids_sum = sum(grids,3);

figure;
imagesc(grids_sum);
axis equal;
axis tight;
colorbar;

% grids_weighted = nan(4,4,n_bases);
% for i = 1:n_bases
%     grids_weighted(:,:,i) = grids(:,:,i).*weights(i);
% end
% weighted_mean = mean(grids_weighted,3);
