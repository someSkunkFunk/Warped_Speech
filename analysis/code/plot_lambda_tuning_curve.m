function best_lam=plot_lambda_tuning_curve(stats_obs,trf_config,plot_ch)
arguments
    stats_obs (1,1) struct
    trf_config (1,1) struct
    plot_ch (1,1) double=85; % note: could extend functionality easily to plotting multiple chns
end
%TODO: prettify plot paramms
%plot tuning curve based on null_distribution test based on all conditions
%trf
% avg across trials
R_curves=squeeze(mean(stats_obs.r,1));
% [~,best_lam_idx]=max(R_curves,[],1);
% 
% [~,best_chn_idx]=max(R_curves,[],2);
[r_max, r_max_idx]=max(R_curves,[],'all');
[best_lam_idx,best_chn_idx]=ind2sub(size(R_curves),r_max_idx);
best_lam=trf_config.lam_range(best_lam_idx);
%TODO: consider putting both curves on same plot for comparison
%% plot "best" electrode
tuning_curve_plot_wrapper(R_curves,trf_config.lam_range,best_chn_idx)
hold on
xline(trf_config.lam_range(best_lam_idx),'r--')
temp_tit=sprintf('best channel (%d)',best_chn_idx);
legend('tuning curve',sprintf('max r: %0.2g, \\lambda:%0.2g',r_max,best_lam))
title(temp_tit), clear temp_tit, hold off

%% plot another particular electrode
[tc_max_r, tc_max_lam_idx]=max(R_curves(:,plot_ch));
tuning_curve_plot_wrapper(R_curves,trf_config.lam_range,plot_ch)
hold on
stem(trf_config.lam_range(best_lam_idx),r_max,'r--');
stem(trf_config.lam_range(tc_max_lam_idx),tc_max_r,'r--')
legend('tuning curve',sprintf('max r: %0.2g, \\lambda:%0.2g',r_max,best_lam), ...
    sprintf('curve max r: %0.2g, \\lambda:%0.2g',tc_max_r,trf_config.lam_range(tc_max_lam_idx)))

temp_tit=sprintf('Chn # (%d) tuning curve',plot_ch);
title(temp_tit), clear temp_tit, hold off

function tuning_curve_plot_wrapper(R_curves,lam_vals,plot_chn)
tuning_curve=R_curves(:,plot_chn);
figure
plot(lam_vals,tuning_curve)
set(gca(),'XScale','log')
xlabel('lambda')
ylabel('mean(r) across trials')


end
end