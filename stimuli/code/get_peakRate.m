function [peakRate, env, env_onsets,env_thresh]=get_peakRate(env,fs,env_thresh_std)
% [peakTs,peakVals,p,w]=get_peakRate(env,fs,warp_config)


% [peakTs,peakVals]=peakRate(env,fs,peak_tol)
% TODO: can we make Hd an input param conveniently? 
Hd = getLPFilt(fs,10); %% Maybe don't filter so harshly?
env = filtfilthd(Hd,env);
% rectify + remove noisy envelope fluctuations to imrpove peakrate algo
env_thresh=std(env)*env_thresh_std;
env(env<env_thresh)=0;
% Find onsets
env_onsets = diff(env);
%only case about positive peaks
env_onsets(env_onsets<0) = 0;

[pkVals,pkTimes,w,p] = findpeaks(env_onsets,fs);
% fprintf('note: calculating peakwidth using both references... remove this feature before running on final warp script')
% [~,~,w2,~] = findpeaks(env_onset,fs,'WidthReference','halfheight');
% normalize prominence
% NOTE: 'normalizing' will still leave values above 1
% NOTE: maybe not necessary...?

p = p./std(p);
w = w./std(w);

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