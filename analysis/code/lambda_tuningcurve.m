%%%%%%%%%%%%%%%%%%% DO NOT USE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%plot tuning curve based on null_distribution test based on all conditions
%trf
clear,clc,close all 
subj=19;
% kind of a misnomer here... should have already computed optimization
% before trying to plot lambda curve, but we need the null_distribution
% file

%^TODO: verify if above is true since updated code doesn't generate a
%null_distribution file?
do_lambda_optimization=true;

preprocess_config=config_preprocess(subj);
trf_config=config_trf(subj,do_lambda_optimization,preprocess_config);
% load(trf_config.nulldistribution_file);
load(trf_config.model_metric_path)
% error checking that will be useful as we continue refactoring code:
% if all(trf_config.lam_range==lam)
%     best_lambda=trf_config.lam_range(best_lam_idx);
% else
%     error('u done fuked up.')
% end
% ^ not sure what the purpose of these lines was...

% average r across trials
best_lam_idx=find(trf_config.lam_range==trf_config.best_lam);
r_avg_best_lam=squeeze(mean(stats_obs.r(:,best_lam_idx,:),1));
%TODO: get this from trf_analysis script instead of here
[r_max,best_chn_idx]=max(r_avg_best_lam);
fprintf('best_channel (%d) has max_r of %0.3f\n',best_chn_idx,r_max);

all_chn_tuning_curves=squeeze(mean(stats_obs.r,1));
%TODO: consider putting both curves on same plot for comparison
%% plot "best" electrode
tuning_curve_plot_wrapper(all_chn_tuning_curves,trf_config.lam_range,best_chn_idx)
temp_tit=sprintf('best channel (%d) tuning curve',best_chn_idx);
title(temp_tit), clear temp_tit

%% plot another particular electrode
%Fcz or some shit
plot_ch=85;
tuning_curve_plot_wrapper(all_chn_tuning_curves,trf_config.lam_range,plot_ch)
temp_tit=sprintf('other channel (%d) tuning curve',plot_ch);
title(temp_tit), clear temp_tit
% doesn't seem like stats_obs.r needs squeeze???
% dum=squeeze(stats_obs.r);
% dum_ch=dum(:,:,plot_ch);
% meanR_dm_ch=mean(dum_ch,1);
% figure
% plot(lam,meanR_dm_ch)
% set(gca(),'XScale','log')
% xlabel('lambda')
% ylabel('mean(r) across trials')
% 
% title_str=sprintf('subj %d lambda tuning curve electrode %d - bpfilter: %.2g-%.2g Hz' ...
%     ,subj,plot_ch,bpfilter(1),bpfilter(2));
% title(title_str)
% [~,maxIndx]=max(meanR_dm_ch);
% fprintf('subj %d chn %d max lambda:%0.2g\n',subj,plot_ch,lam(maxIndx))

function tuning_curve_plot_wrapper(all_chn_tuning_curves,lam_vals,plot_chn)
tuning_curve=all_chn_tuning_curves(:,plot_chn);
figure
plot(lam_vals,tuning_curve)
set(gca(),'XScale','log')
xlabel('lambda')
ylabel('mean(r) across trials')


end
%% BANISHED REALM

% if do_correction
%     corrstr='with';
% else
%     corrstr='without';
% end
% for separated conditions
% cc=1; %which condition
% dum=squeeze(stats_obs{cc}.r);
% dum_ch=dum(:,:,plot_ch);
% figure
% plot(lam,mean(dum_ch,1))
% set(gca(),'XScale','log')
% xlabel('lambda')
% ylabel('mean(r) across trials')
% if do_correction
%     corrstr='with';
% else
%     corrstr='without';
% end
% title_str=sprintf('lambda tuning curve electrode %d %s correction',plot_ch,corrstr);
% title(title_str)