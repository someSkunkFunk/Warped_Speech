function [peakRate, env, diff_env,env_thresh]=get_peakRate(env,fs,warp_config)
% [peakTs,peakVals,p,w]=get_peakRate(env,fs,warp_config)

env_thresh_std=warp_config.env_thresh_std;
% denv_noise_tol=warp_config.env_derivative_noise_tol;
min_pkrt_height=warp_config.min_pkrt_height;
% [peakTs,peakVals]=peakRate(env,fs,peak_tol)
% TODO: can we make Hd an input param conveniently? 
Hd = getLPFilt(fs,10); %% Maybe don't filter so harshly?
env = filtfilthd(Hd,env);
% rectify + remove noisy envelope fluctuations (from lowpass filtering)
env_thresh=std(env)*env_thresh_std;
env_noise_mask=env<env_thresh;
env(env_noise_mask)=0;
% Find onsets
diff_env = [diff(env) 0];
%only care about positive peaks
diff_env(diff_env<0) = 0;
% remove spurious peaks envelope fluctuations that persist in quiet periods
% due to filtering
diff_env(env_noise_mask(1:end-1))=0;

[pkVals,pkTimes,w,p] = findpeaks(diff_env,fs,'MinPeakHeight',min_pkrt_height);
% fprintf('note: calculating peakwidth using both references... remove this feature before running on final warp script')
% [~,~,w2,~] = findpeaks(env_onset,fs,'WidthReference','halfheight');
% normalize prominence
% NOTE: 'normalizing' will still leave values above 1
% NOTE: maybe not necessary...?

% p = p./std(p);
% w = w./std(w);

peakRate=struct('pkVals',pkVals,'pkTimes',pkTimes,'p',p,'w',w);
% w2 = w2./std(w2);

% Eliminate small peaks
% % (note that if peak_tol is zero, this will remove nothing since prominence is nonzero)
% peakTs(p<peak_tol)=[];
% peakVals(p<peak_tol)=[];
% w(p<peak_tol)=[];
% % w2(p<peak_tol)=[];
% p(p<peak_tol)=[];
end