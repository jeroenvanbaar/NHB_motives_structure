global Ss Ts softmax
Ss = [0,3,7,10];
Ts = [5,8,12,15];
softmax = @(values, inv_temp) exp(values.*inv_temp)./sum(exp(values.*inv_temp));
