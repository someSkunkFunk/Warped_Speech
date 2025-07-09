function [peakTs,peakVals,p,w,w2]=get_peakRate(env,fs,peak_tol)
% [peakTs,peakVals,p,w]=get_peakRate(env,fs,peak_tol)

arguments
    env
    fs
    % Hd % NOTE: why do we have it as an input param but it's defined within the function??
    peak_tol (1,1) double = 0;
end
% [peakTs,peakVals]=peakRate(env,fs,peak_tol)
% TODO: can we make Hd an input param conveniently? 
Hd = getLPFilt(fs,10); %% Maybe don't filter so harshly?
env = filtfilthd(Hd,env);
% Find onsets
env_onset = diff(env);
env_onset(env_onset<0) = 0;

[peakVals,peakTs,w,p] = findpeaks(env_onset,fs);
fprintf('note: calculating peakwidth using both references... remove this feature before running on final warp script')
[~,~,w2,~] = findpeaks(env_onset,fs,'WidthReference','halfheight');
% normalize prominence
% NOTE: 'normalizing' will still leave values above 1
% NOTE: maybe not necessary...?

p = p./std(p);
w = w./std(w);
w2 = w2./std(w2);

% Eliminate small peaks
peakTs(p<peak_tol)=[];
peakVals(p<peak_tol)=[];
w(p<peak_tol)=[];
w2(p<peak_tol)=[];
p(p<peak_tol)=[];
end