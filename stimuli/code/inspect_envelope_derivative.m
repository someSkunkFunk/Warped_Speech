function inspect_envelope_derivative(diff_env,peakRate,fs,warp_config)
% inspect_segs(wf,fs,Ifrom,seg,env)
%plot envelope derivative along with prominence + threshold values
t_vec=0:1/fs:(length(diff_env)-1)/fs;
% boost small peaks so they become visible?
% histogram env_onsets to see if there's a clear point where we can
% threshold?
figure, h=histogram(peakRate.pkVals,'Normalization','count');
n_peaks=sum(peakRate.p>warp_config.prom_thresh& ...
    peakRate.w>warp_config.width_thresh& ...
    peakRate.p.*peakRate.w>warp_config.area_thresh);
N_peaks=numel(peakRate.pkTimes);

ylim([0 N_peaks])
title(sprintf('peakRate.vals histogram - max(counts): %d; n_{peaks}/N: %d/%d',max(h.Values),n_peaks,N_peaks))
figure
ax1=subplot(4,1,1);
plot(t_vec,diff_env,"Color",'k')
hold on
plot([min(t_vec) max(t_vec)], ones(1,2).*warp_config.min_pkrt_height,'r--')
ylim([0 max(diff_env)])
legend('envelope derivative',sprintf('min pk height:%0.3g',warp_config.min_pkrt_height))
ax2=subplot(4,1,2);
stem(peakRate.pkTimes,peakRate.w)
hold on
plot([min(t_vec) max(t_vec)],warp_config.width_thresh*ones(2,1),'r--')
legend('peakwidth', sprintf('thresh: %0.3g',warp_config.width_thresh))
ax3=subplot(4,1,3);
stem(peakRate.pkTimes,peakRate.p)
hold on
plot([min(t_vec) max(t_vec)],warp_config.prom_thresh*ones(2,1),'r--')
legend('prominence',sprintf('thresh: %0.3g',warp_config.prom_thresh))
ax4=subplot(4,1,4);
stem(peakRate.pkTimes,peakRate.w.*peakRate.p)
hold on
plot([min(t_vec) max(t_vec)],warp_config.prom_thresh*ones(2,1),'r--')
legend('prominence*width',sprintf('thresh: %0.3g',warp_config.area_thresh))
xlabel('Time (s)')
linkaxes([ax1 ax2 ax3 ax4],'x')
% plot histograms of prominence, peakwidth, and peak area

figure
subplot(3,1,1)
histogram(peakRate.p,'Normalization','count')
title('Peak Prominence')
subplot(3,1,2)
histogram(peakRate.w,'Normalization','count')
title('peakwidth')
subplot(3,1,3)
histogram(peakRate.w.*peakRate.p,'Normalization','count')
title('peak area')
sgtitle(sprintf('%s env, n_{peaks}/N: %d/%d',warp_config.env_method,n_peaks,N_peaks))

end