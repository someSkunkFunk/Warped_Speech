%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NOTES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%configs are saved without paths but not loaded during script - instead
%those initialized here are used and they contain the paths.
%load_checkpoint uses saved config to match to correct file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NOTES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

overwrite=false;
% when config fields are structured, run with option below as true - will
% igonore overwrite (assumes saved file exists)
update_configs=true;

%%%%%%%%%%%%%%% params that we mostly change in fast/slow %%%%%%%%%%%%%%%%%

preprocess_config.subj=subj;

trf_config.separate_conditions=false;
trf_config.crossvalidate=true; %note: i think the intended behavior when 
% this is false hasn't been properly programmed into the analysis script
% logic partially because I'm not sure what kind of behavior we want but
% probably it's just to train a model with a particular hard-coded lam 
% value in which case we'd set lam_range as well probably...?

do_nulltest=true;

%%%%%%%%%%%%%%%%%%%%%%% additional params whose logic we've%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% automated in fast/slow %%%%%%%%%%%%%%%%%%%%%%%%%%%%
if subj<=90
    preprocess_config.experiment='fast-slow';
else
    preprocess_config.experiment='reg-irreg';
end
% use extended timelims for model so TRFs don't include edge artefacts

%note: it is my understanding that filter-generated ringing artefacts won't
%leak into the TRF because enough time has passed before recording unpause
%and stimulus onset for trial and trials are epoched from the time the
%sound starts not the unpause of recording... 

%TODO:
% warning(['reminder that train_params should be included in trf config in ' ...
%     'future, bypassing for now to continue progress..'])
% train_params.tmin_ms=-500; 
% train_params.tmax_ms=800;

if ~trf_config.separate_conditions
    trf_config.do_lambda_optimization=true;
else
    % we assume optimization done data from all conditions and get best
    % lambda from saved checkpoint later
    trf_config_=trf_config;
    trf_config_.separate_conditions=false;
    trf_config_.do_lambda_optimization=true;
end


%%%%%%%%%%%%%%%%%%%%%%%%% sanity checks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% if doing nulltest, we also need stats_obs from cross validate regardless
% of conditions separate or not
if do_nulltest && ~trf_config.crossvalidate
    warning(['trf_config.crossvalidate set to false with do_nulltest set\n' ...
        'to true - need crossvalidation to generate nulldist so changing\n' ...
        'trf_config.crossvalidate to true'])
    trf_config.crossvalidate=true;
end


%%%% INSTANTIATE ALL CONFIGS LAST SO SET PARAMS DONT GET OVERWRITTEN BY DEFAULTS &&&&&&&&
preprocess_config=config_preprocess(preprocess_config);

if exist("trf_config_","var")
    % load best_lambda from condition-agnostic crossvalidation with same
    % preprocessing params
    trf_config_=config_trf(trf_config_,preprocess_config);
    S_=load_checkpoint(trf_config_);
    trf_config.train_params.best_lam=plot_lambda_tuning_curve(S_.stats_obs,S_.config);
    clear trf_config_ S_
end

trf_config=config_trf(trf_config,preprocess_config);
