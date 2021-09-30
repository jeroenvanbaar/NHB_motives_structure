%% Define basis set

function [basis_set,combs] = define_basis_set(features)
    
    % The value in each S,T element indicates whether this basis set predicts
    % cooperate (logical 1) or defect (0) for this game
    basis_set = struct('name',[],'model',[]);
    if regexp(features,'Co')
        basis_set(length(basis_set)+1).name = 'Coop';
        basis_set(length(basis_set)).model = @(S,T) 1;
    end
    if regexp(features,'De')
        basis_set(length(basis_set)+1).name = 'Defect';
        basis_set(length(basis_set)).model = @(S,T) 0;
    end
    if regexp(features,'Gr')
        basis_set(length(basis_set)+1).name = 'Greed';
        basis_set(length(basis_set)).model = @(S,T) max([10, S]) - max([T, 5]) > 0; % Max-max
    end
    if regexp(features,'Ig')
        basis_set(length(basis_set)+1).name = 'InverseGreed';
        basis_set(length(basis_set)).model = @(S,T) max([10, S]) - max([T, 5]) <= 0; % Max-max
    end
    if regexp(features,'Ri')
        basis_set(length(basis_set)+1).name = 'Risk';
        basis_set(length(basis_set)).model = @(S,T) min([10, S]) - min([T, 5]) > 0; % Max-min
    end
    if regexp(features,'Ir')
        basis_set(length(basis_set)+1).name = 'InverseRisk';
        basis_set(length(basis_set)).model = @(S,T) min([10, S]) - min([T, 5]) <= 0; % Max-min
    end
    if regexp(features,'Re')
        basis_set(length(basis_set)+1).name = 'Regret';
        basis_set(length(basis_set)).model = @(S,T) mean([(T-5) - (10-S) >= 0, ...
            (T-5) - (10-S) > 0]);
    end
    if regexp(features,'Na')
        basis_set(length(basis_set)+1).name = 'Nash';
        basis_set(length(basis_set)).model = @(S,T) mean([S >= (T - 5), ...
            S > (T - 5)]);
    end
    if regexp(features,'En')
        basis_set(length(basis_set)+1).name = 'Envy';
        basis_set(length(basis_set)).model = @(S,T) S >= T;
    end
    if regexp(features,'Ss')
        basis_set(length(basis_set)+1).name = 'S';
        basis_set(length(basis_set)).model = @(S,T) S/10;
    end
    if regexp(features,'Ts')
        basis_set(length(basis_set)+1).name = 'T';
        basis_set(length(basis_set)).model = @(S,T) (T-5)/10;
    end
    if regexp(features,'Hg')
        basis_set(length(basis_set)+1).name = 'Harmony';
        basis_set(length(basis_set)).model = @(S,T) (S > 5) & (T < 10);
    end
    if regexp(features,'Sh')
        basis_set(length(basis_set)+1).name = 'StagHunt';
        basis_set(length(basis_set)).model = @(S,T) (S < 5) & (T < 10);
    end
    if regexp(features,'Sg')
        basis_set(length(basis_set)+1).name = 'Snowdrift';
        basis_set(length(basis_set)).model = @(S,T) (S > 5) & (T > 10);
    end
    if regexp(features,'Pd')
        basis_set(length(basis_set)+1).name = 'Prisoners';
        basis_set(length(basis_set)).model = @(S,T) (S < 5) & (T > 10);
    end
    if regexp(features,'ST')
        global Ss Ts
        for Si = Ss
            for Ti = Ts
                basis_set(length(basis_set)+1).name = sprintf('S%iT%i',Si,Ti);
                basis_set(length(basis_set)).model = @(S,T) (S == Si) & (T == Ti);
            end
        end
    end
    basis_set = basis_set(2:end);
    basis_set_names = cell(length(basis_set),1);
    for bfi = 1:length(basis_set)
        basis_set_names{bfi} = basis_set(bfi).name;
    end
    n_features_full_basis_set = length(basis_set);

    % All sets of feature combinations
    n_combs = 2^n_features_full_basis_set - 1;
    combs = sortrows(dec2bin(1:n_combs));
    
end