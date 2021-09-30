%% Load game data

function [gaze_dat, sub_IDs] = load_gaze_dat
    opts = detectImportOptions('../all_fixations_as_pct_LT_eye-average.csv');
    opts = setvartype(opts,{'sub','part','block','trial','dur_pct'},'single');
    opts = setvartype(opts,{'player_type','num','num_S_T'},'string');
    clear gaze_dat
    gaze_dat = readtable('../all_fixations_as_pct_LT_eye-average.csv',opts,...
        'ReadRowNames',false);
%     sub_ID_fun = @(x) string(sprintf('50%02d',x));
%     gaze_dat.subID = varfun(sub_ID_fun, gaze_dat(:,'sub'));
    sub_IDs = unique(gaze_dat.sub);
end