overwrite=false;
preprocess_config.subj=subj;
trf_config.subj=subj;
trf_config.separate_conditions=true;
do_nulltest=true;
trf_config.crossvalidate=true; %note: i think the intended behavior when 
% this is false hasn't been properly programmed into the analysis script
% logic partially because I'm not sure what kind of behavior we want but
% probably it's just to train a model with a particular hard-coded lam 
% value in which case we'd set lam_range as well probably...?

% use extended timelims for model so TRFs don't include edge artefacts

%note: it is my understanding that filter-generated ringing artefacts won't
%leak into the TRF because enough time has passed before recording unpause
%and stimulus onset for trial and trials are epoched from the time the
%sound starts not the unpause of recording... 
train_params.tmin_ms=-500; 
train_params.tmax_ms=800;
if ~trf_config.separate_conditions
    trf_config.do_lambda_optimization=true;
    % do full crossvalidation only to get optimal lambda in
    % condition-agnostic way
    % we need to crossvalidate to optimize lambda
    % in this case, we only use causal trf window that is shorter to speed
    % up process
    % trf_config.tmin_ms=0;
    % trf_config.tmin_ms=400;
else
    % we assume optimization done data from all conditions and get best
    % lambda from saved checkpoint later
    trf_config_=trf_config;
    trf_config_.separate_conditions=false;
    trf_config_.do_lambda_optimization=true;
end
% if doing nulltest, we also need stats_obs from cross validate regardless
% of conditions separate or not
if do_nulltest && ~trf_config.crossvalidate
    warning(['trf_config.crossvalidate set to false with do_nulltest set\n' ...
        'to true - need crossvalidation to generate nulldist so changing\n' ...
        'trf_config.crossvalidate to true'])
    trf_config.crossvalidate=true;
end

preprocess_config=config_preprocess(preprocess_config);
trf_config=config_trf(trf_config,preprocess_config);
if exist("trf_config_","var")
    % load best_lambda from condition-agnostic crossvalidation with same
    % preprocessing params
    trf_config_=config_trf(trf_config_,preprocess_config);
    S_=load_checkpoint(trf_config_);
    train_params.best_lam=plot_lambda_tuning_curve(S_.stats_obs,S_.config);
    clear trf_config_ S_
end