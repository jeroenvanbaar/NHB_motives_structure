%% Define basis set
% These bases contain 1s (cooperate), -1s (defect), or 0s (no prediction)

function [basis_set,combs] = define_basis_set_2(features)
    
    % The value in each S,T element indicates whether this basis set predicts
    % cooperate (logical 1) or defect (-1) for this game
    basis_set = struct('name',[],'model',[]);
    if regexp(features,'Co')
        basis_set(length(basis_set)+1).name = 'Coop';
        basis_set(length(basis_set)).model = @(S,T) 1;
    end
    if regexp(features,'De')
        basis_set(length(basis_set)+1).name = 'Defect';
        basis_set(length(basis_set)).model = @(S,T) -1;
    end
    if regexp(features,'Gr')
        basis_set(length(basis_set)+1).name = 'Greed';
        basis_set(length(basis_set)).model = @(S,T) 2*(max([10, S]) - max([T, 5]) > 0)-1; % Max-max
    end
    if regexp(features,'Ig')
        basis_set(length(basis_set)+1).name = 'InverseGreed';
        basis_set(length(basis_set)).model = @(S,T) 2*(max([10, S]) - max([T, 5]) <= 0)-1; % Max-max
    end
    if regexp(features,'Ri')
        basis_set(length(basis_set)+1).name = 'Risk';
        basis_set(length(basis_set)).model = @(S,T) 2*(min([10, S]) - min([T, 5]) > 0)-1; % Max-min
    end
    if regexp(features,'Ir')
        basis_set(length(basis_set)+1).name = 'InverseRisk';
        basis_set(length(basis_set)).model = @(S,T) 2*(min([10, S]) - min([T, 5]) <= 0)-1; % Max-min
    end
    if regexp(features,'Re')
        basis_set(length(basis_set)+1).name = 'Regret';
        basis_set(length(basis_set)).model = @(S,T) 2*(mean([(T-5) - (10-S) >= 0, ...
            (T-5) - (10-S) > 0]))-1;
    end
    if regexp(features,'Na')
        basis_set(length(basis_set)+1).name = 'Nash';
        basis_set(length(basis_set)).model = @(S,T) 2*(mean([S >= (T - 5), ...
            S > (T - 5)]))-1;
    end
    if regexp(features,'EV')
        basis_set(length(basis_set)+1).name = 'ExpVal';
        basis_set(length(basis_set)).model = @(S,T) 2*(mean([S >= (T - 5), ...
            S > (T - 5)]))-1;
    end
    if regexp(features,'En')
        basis_set(length(basis_set)+1).name = 'Envy';
        basis_set(length(basis_set)).model = @(S,T) 2*(S >= T)-1;
    end
    if regexp(features,'Ss')
        basis_set(length(basis_set)+1).name = 'S';
        basis_set(length(basis_set)).model = @(S,T) 2*(S/10)-1;
    end
    if regexp(features,'Ts')
        basis_set(length(basis_set)+1).name = 'T';
        basis_set(length(basis_set)).model = @(S,T) 2*((T-5)/10)-1;
    end
    % THE FOLLOWING GAME TYPE BASES ARE AGNOSTIC ABOUT GAMES
    % OUTSIDE THE GAME TYPE - no generalization across game types
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
    % THE FOLLOWING GAME BASES ONLY MAKE PREDICTIONS FOR ONE S,T SQUARE
    % (no generalization at all)
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