%% Create 1000 random basis sets to use

n_random_basis_sets = 5000;
n_random_bases_per_set = 4;
random_bases = nan(n_random_basis_sets,n_random_bases_per_set,4,4);
for random_basis_set_i = 1:n_random_basis_sets
    
    if mod(random_basis_set_i,10) == 0
        disp(random_basis_set_i);
    end
    % Define random basis functions
%     random_bases = zeros(4,4,n_random_bases_per_set);
    for random_basis_i = 1:n_random_bases_per_set
        if random_basis_i == 1
            random_basis = Create_random_basis_set;
            random_bases(random_basis_set_i,random_basis_i,:,:) = random_basis;
%             random_bases(:,:,random_basis_i) = random_basis;
        else
            aok = false;
            n_to_check = random_basis_i - 1; % Compare to all previous ones
            while aok == false
                random_basis = Create_random_basis_set; % Create
                sim_scores = zeros(1,n_to_check); % Preallocate checks
                for check_i = 1:n_to_check
                    sim_scores(check_i) = sum(sum(random_basis == ...
                        reshape(random_bases(random_basis_set_i,check_i,:,:),4,4)));
                end
                if sum(sim_scores < 12) == n_to_check
                    aok = true;
%                     random_bases(:,:,random_basis_i) = random_basis;
                    random_bases(random_basis_set_i,random_basis_i,:,:) = random_basis;
                end
            end
        end
    end
%     random_bases_1000(random_basis_set_i,:,:,:) = random_bases;
end

%% Store
save(sprintf('random_bases_%i.mat',n_random_basis_sets),'random_bases')

%% Create 1000 random basis sets as 'shuffled' versions of psychologically motivated bases

n_random_basis_sets = 5000;
n_random_bases_per_set = 4;
random_bases = nan(n_random_basis_sets,n_random_bases_per_set,4,4);
for random_basis_set_i = 1:n_random_basis_sets
    
    if mod(random_basis_set_i,10) == 0
        disp(random_basis_set_i);
    end
    % Define random basis functions
    % Start with coop basis
    random_bases(random_basis_set_i,1,:,:) = ones(4,4);
    
    % Add two based on greed/risk
    for random_basis_i = 2:3
        aok = false;
        n_to_check = random_basis_i - 1;
        while aok == false
            random_basis = Create_random_basis_set;
            sim_scores = zeros(1,n_to_check);
            for check_i = 1:n_to_check
                sim_scores(check_i) = sum(sum(random_basis == ...
                    reshape(random_bases(random_basis_set_i,check_i,:,:),4,4)));
            end
            if sum(sim_scores < 12) == n_to_check
                aok = true;
                random_bases(random_basis_set_i,random_basis_i,:,:) = random_basis;
            end
        end
    end

    % Add one based on regret
    for random_basis_i = 4
        aok = false;
        n_to_check = random_basis_i - 1;
        while aok == false
            random_basis = Create_random_basis_regret;
            sim_scores = zeros(1,n_to_check);
            for check_i = 1:n_to_check
                sim_scores(check_i) = sum(sum(random_basis == ...
                    reshape(random_bases(random_basis_set_i,check_i,:,:),4,4)));
            end
            if sum(sim_scores < 12) == n_to_check
                aok = true;
                random_bases(random_basis_set_i,random_basis_i,:,:) = random_basis;
            end
        end
    end
end

%% Plot one
figure;
random_basis_set_i = 4;
for i = 1:4
    subplot(1,4,i);
    imagesc(reshape(random_bases(random_basis_set_i,i,:,:),4,4));
    axis equal
    axis tight
    caxis([0,1]);
end

%% Store
save(sprintf('random_bases_%i_shuffleTrueBasisSet.mat',n_random_basis_sets),'random_bases');






