%% Feature RL and Bayesian models:
results = [];
skipi = 0;
for smi = 1:5
    for sim_index = 1:100
        for mfsi = 1:5
            tmp = load(sprintf('results/results_smi-%i_sim-%i_mfsi-%i.mat',...
                smi,sim_index,mfsi));
            SSE = tmp.results_mat.SSE;
            LL = tmp.results_mat.LL;
            BIC = tmp.results_mat.BIC;
            try
                AIC = tmp.results_mat.AIC;
            catch
                AIC = nan;
                skipi = skipi + 1;
                fprintf('No AIC for: smi %i, sim_index %i, mfsi %i (missing value nr %i)\n',smi,sim_index,mfsi,skipi);
            end
            results = [results;[smi,sim_index,mfsi,SSE,LL,AIC,BIC]];
        end
    end
end

%% Count best-fitting models
best_per_sim = [];
missing = [];
var_col = 4;
for smi = 1:5
    for sim_index = 1:100
        cost_per_model_type = results(results(:,1)==smi & results(:,2)==sim_index,var_col);
        if length(cost_per_model_type) ~= 5
            missing = [missing;[smi,sim_index]];
            fprintf('BICs length %i for smi %i and sim_index %i\n',length(cost_per_model_type),smi,sim_index);
        end
        [best_BIC,best_mfsi] = min(cost_per_model_type);
        best_per_sim = [best_per_sim;[smi,sim_index,best_mfsi]];
    end
end

%% Compute proportion recovery
confusion_matrix = nan(5,5);
for smi = 1:5
    for mfsi = 1:5
        confusion_matrix(smi,mfsi) = sum(best_per_sim(:,1)==smi & best_per_sim(:,3)==mfsi);
    end
end

%% Compute inversion matrix

inversion_matrix = confusion_matrix./(sum(confusion_matrix,1)).*100;

%% Plot
model_labels = {'Games','Game Types','Motives','Bayes Motives','Bayes Players'};
figure(1);
subplot(1,2,1);
imagesc(confusion_matrix, [0,100]);
axis equal; axis tight;
title('Confusion matrix: P(recovered|true)');
xticks(1:5); xticklabels(model_labels);
yticks([1,2,3,4,5]); yticklabels(model_labels);
xlabel('Recovered model'); ylabel('True model');
hold on;
for i=1:5
    for j=1:5
        text(j,i,sprintf('%i%%',confusion_matrix(i,j)),...
            'horizontalalignment','center','verticalalignment','middle');
    end
end
cb = colorbar;
set(get(cb,'label'),'string','%');
% Inv matrix
subplot(1,2,2);
imagesc(inversion_matrix, [0,100]);
axis equal; axis tight;
title('Inversion matrix: P(true|recovered)');
xticks(1:5); xticklabels(model_labels);
yticks([1,2,3,4,5]); yticklabels(model_labels);
xlabel('Recovered model'); ylabel('True model');
hold on;
for i=1:5
    for j=1:5
        text(j,i,sprintf('%.0f%%',inversion_matrix(i,j)),...
            'horizontalalignment','center','verticalalignment','middle');
    end
end
cb = colorbar;
set(get(cb,'label'),'string','%');

% print(gcf,'model_recovery.png','-dpng','-r300');




%% *****************************************************
%% *****************************************************
%% *****************************************************

%% feature RL models only:
results = []; skipi = 0;
for smi = 1:3
    for sim_index = 1:100
        for mfsi = 1:3
            tmp = load(sprintf('results/results_smi-%i_sim-%i_mfsi-%i.mat',...
                smi,sim_index,mfsi));
            SSE = tmp.results_mat.SSE;
            LL = tmp.results_mat.LL;
            BIC = tmp.results_mat.BIC;
            try
                AIC = tmp.results_mat.AIC;
            catch
                AIC = nan;
                skipi = skipi + 1;
                fprintf('No AIC for: smi %i, sim_index %i, mfsi %i (missing value nr %i)\n',smi,sim_index,mfsi,skipi);
            end
            results = [results;[smi,sim_index,mfsi,SSE,LL,AIC,BIC]];
        end
    end
end
% Count best-fitting models
best_per_sim = []; missing = [];
var_col = 7;
for smi = 1:3
    for sim_index = 1:100
        cost_per_model_type = results(results(:,1)==smi & results(:,2)==sim_index,var_col);
        if length(cost_per_model_type) ~= 3
            missing = [missing;[smi,sim_index]];
            fprintf('BICs length %i for smi %i and sim_index %i\n',length(cost_per_model_type),smi,sim_index);
        end
        [best_BIC,best_mfsi] = min(cost_per_model_type);
        best_per_sim = [best_per_sim;[smi,sim_index,best_mfsi]];
    end
end
% Compute proportion recovery
confusion_matrix = nan(3,3);
for smi = 1:3
    for mfsi = 1:3
        confusion_matrix(smi,mfsi) = sum(best_per_sim(:,1)==smi & best_per_sim(:,3)==mfsi);
    end
end
% Compute inversion matrix
inversion_matrix = confusion_matrix./(sum(confusion_matrix,1)).*100;

%% Plot
model_labels = {'Games','Game Types','Motives'};
figure(2);
subplot(1,2,1);
imagesc(confusion_matrix, [0,100]);
axis equal; axis tight;
title('Confusion matrix: P(recovered|true)');
xticks(1:3); xticklabels(model_labels);
yticks([1,2,3]); yticklabels(model_labels);
xlabel('Recovered model'); ylabel('True model');
hold on;
for i=1:3
    for j=1:3
        text(j,i,sprintf('%i%%',confusion_matrix(i,j)),...
            'horizontalalignment','center','verticalalignment','middle');
    end
end
cb = colorbar;
set(get(cb,'label'),'string','%');
% Inv matrix
subplot(1,2,2);
imagesc(inversion_matrix, [0,100]);
axis equal; axis tight;
title('Inversion matrix: P(true|recovered)');
xticks(1:3); xticklabels(model_labels);
yticks([1,2,3]); yticklabels(model_labels);
xlabel('Recovered model'); ylabel('True model');
hold on;
for i=1:3
    for j=1:3
        text(j,i,sprintf('%.0f%%',inversion_matrix(i,j)),...
            'horizontalalignment','center','verticalalignment','middle');
    end
end
cb = colorbar;
set(get(cb,'label'),'string','%');

% print(gcf,'model_recovery_RL_only.png','-dpng','-r300');

