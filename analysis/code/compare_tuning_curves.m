%% compate_tuning_curves
%currently, lags are normalized across optimization/TRF model fitting
%steps, so can initialize configs by just running params script
subjs=[2:7,9:22];
best_lam=nan(length(subjs),2); % [subjects x before-after]
for ss=1:length(subjs)
    subj=subjs(ss);
    script_config=struct('show_tuning_curves',false);
    trf_analysis_params;   
    trf_config_consistent_lags=trf_config;
    trf_config_inconsistent_lags=trf_config; clear trf_config
    trf_config_inconsistent_lags.tmax_ms=400;
    trf_config_inconsistent_lags.tmin_ms=0;
    
    
    %% load stats_obs consisent  & inconsistent
    S_inconsistent=load_checkpoint(trf_config_inconsistent_lags);
    S_consistent=load_checkpoint(trf_config_consistent_lags);
    
    
    %%
    best_lam(ss,1)=plot_lambda_tuning_curve(S_inconsistent.stats_obs,trf_config_inconsistent_lags);
    best_lam(ss,2)=plot_lambda_tuning_curve(S_consistent.stats_obs,trf_config_consistent_lags);
    clear S_inconsistent S_consistent trf_config_inconsistent_lags trf_config_consistent_lags
end
%%
figure("Name","Ridge Parameter Change From Normalizing Lag Windows", ...
    "Color","w")
plot(subjs,best_lam(:,1),'DisplayName','Before'), hold on
plot(subjs,best_lam(:,2),'DisplayName','After')
xlabel('subj num')
ylabel('ridge parameter')
title('Effect of Normalizing Lag Windows')
legend()


%% --- plot change across subjects --- 

