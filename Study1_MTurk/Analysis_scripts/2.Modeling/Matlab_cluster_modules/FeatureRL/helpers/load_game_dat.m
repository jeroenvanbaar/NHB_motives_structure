%% Load game data

function [game_dat, sub_IDs] = load_game_dat
    opts = detectImportOptions('../gameDat_total.csv');
    opts = setvartype(opts,{'Block','Trial','S','T',...
        'ConfidenceNum','ScoreNum'},'single');
    opts = setvartype(opts,{'subID','Player', 'Type','Variant',...
        'Type_Total','GameType','Colors','GivenAns','CorrAns',...
        'SelfReport'},'string');
    clear game_dat
    game_dat = readtable('../gameDat_total.csv',opts,'ReadRowNames',false);
    game_dat.Properties.VariableNames{16} = 'Confidence';
    game_dat.Properties.VariableNames{17} = 'Score';
    game_dat{game_dat.Trial<8,'Round'} = 1;
    game_dat{game_dat.Trial<4,'Round'} = 0;
    game_dat{game_dat.Trial>7,'Round'} = 2;
    game_dat{game_dat.Trial>11,'Round'} = 3;
    game_dat{:,'Phase'} = 0; % == 'Early'
    game_dat{game_dat.Round>1,'Phase'} = 1; % == 'Late'
    sub_IDs = unique(game_dat.subID);
end