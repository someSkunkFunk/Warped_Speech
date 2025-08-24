% analyze_trfs
% run after plot_trfs to load them first
clearvars -except ind_models avg_models trf_config subjs 
clc
%% Compute PSDS
psd_method='pwelch';
time_range=find(avg_models(1).t>0);
ns=length(time_range);
% individual models
% stack all trfs into 2d mat for efficiency
fs=ind_models(1).fs;
% stack all subjects into a single matrix
[n_subjs,n_cond]=size(ind_models);
[~,~,ne]=size(ind_models(1).w);
if mod(ns,8)~=0
    % only true if we use default
    disp('warning: pwelch will truncate signals when getting psd.')
end

M=nan(ns,n_subjs,n_cond,ne);
for ss=1:n_subjs
    for cc=1:n_cond
        M(:,ss,cc,:)=ind_models(ss,cc).w(1,time_range,:);
    end
end
X=reshape(M,ns,[]);
% [psds,freqs]=periodogram(X,hamming(ns),ns,fs);
[psds,freqs]=get_psd(X,fs,psd_method);
% note: first dimension should be half the number of samples plus 1
P=reshape(psds,[],n_subjs,n_cond,ne);
% average out the subjects
P_mean=squeeze(mean(P,2));

% subject-averaged models
M_avg=nan(ns,n_cond,ne);
for cc=1:n_cond
    M_avg(:,cc,:)=avg_models(cc).w(1,time_range,:);
end
X_avg=reshape(M_avg,ns,[]);
% [psds_avg,~]=periodogram(X_avg,hamming(ns),ns,fs);
[psds_avg,~]=get_psd(X_avg,fs,psd_method);
P_savg=reshape(psds_avg,[],n_cond,ne);


%% plot all individual PSD(TRF) together 

xlims=[0,15];
ylims1=[0 0.01];

figure
plot(freqs,psds)
title('trf psds for all subjects/electrodes')
xlabel('frequency (Hz)')
set(gca(),'XLim',xlims);
%% plot avg(PSDs) 
% sorted by condition
for cc=1:n_cond
    figure
    plot(freqs,squeeze(P_mean(:,cc,:)))
    xlabel('frequency (Hz)')
    title(sprintf('avg(PSD(TRFs)) - condition: %d',cc))
    set(gca(),'XLim',xlims,'YLim',ylims1)
end
% plot subject-averaged, sorted by condition, for a particular electrode
show_single_elec=false;
if show_single_elec
    show_elec=85;
    for cc=1:n_cond
        figure
        plot(freqs,squeeze(P_mean(:,cc,show_elec)))
        xlabel('frequency (Hz)')
        title(sprintf('avg(PSD(TRFs)) condition,electrode: %d,%d',cc,show_elec))
        set(gca(),'XLim',xlims,'YLim',ylims1)
    end
end
%% plot PSD(avg(TRFs)) 
ylims2=[0,.006];

% sorted by condition
for cc=1:n_cond
    figure
    plot(freqs,squeeze(P_savg(:,cc,:)))
    xlabel('frequency (Hz)')
    title(sprintf('PSD(avg(TRFs)) - condition: %d',cc))
    set(gca(),'XLim',xlims,'YLim',ylims2)
end
% plot subject-averaged, sorted by condition, for a particular electrode
show_single_elec=false;
if show_single_elec
    show_elec=85;
    for cc=1:n_cond
        figure
        plot(freqs,squeeze(P_savg(:,cc,show_elec)))
        xlabel('frequency (Hz)')
        title(sprintf('PSD(avg(TRFs)) condition,electrode: %d,%d',cc,show_elec))
        set(gca(),'XLim',xlims,'YLim',ylims2)
    end
end
    %% compare PSDs post-onset only against entire timespan 

%% look at pre-onset PSDs

%% helpers
function [psd,freqs]=get_psd(X,fs,method)
% assumes X is 2D matrix that is [time, waveforms] shape

ns=size(X,1);
win_len=round(ns/3);
% win_len=[];
% remove mean
X=detrend(X,'constant');
switch method
    case 'periodogram'
        [psd,freqs]=periodogram(X,hamming(ns),ns,fs);
    case 'pwelch'
        %TODO: explore effect of changing window size and noverlap...
        [psd,freqs]=pwelch(X,win_len,[],ns,fs);
end
end