%optimize sl_env
% intended use: called from modelSL -- not run on it's own

% test-run to see if we can access functions when running from modelSL
sl_param.f_nat=4; % in Hz -> converted to radians when running model
sl_param.lambda=0.1;
sl_param.gamma=13.83;
sl_param.k=80;
% n_test=
[t, nostim_out]=run_sl_model(config,sl_param);
