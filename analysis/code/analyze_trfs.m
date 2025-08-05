% analyze_trfs
% run after plot_trfs to load them first
clearvars -except ind_models avg_models trf_config subjs 
clc
%% Compute PSDS

% individual models
% stack all trfs into 2d mat for efficiency
fs=ind_models(1).fs;
% stack all subjects into a single matrix
[n_subjs,n_cond]=size(ind_models);
[~,ns,ne]=size(ind_models(1).w);
M=nan(ns,n_subjs,n_cond,ne);
for ss=1:n_subjs
    for cc=1:n_cond
        M(:,ss,cc,:)=ind_models(ss,cc).w;
    end
end
X=reshape(M,ns,[]);
[psds,freqs]=periodogram(X,rectwin(ns),ns,fs);
% note: first dimension should be half the number of samples plus 1
P=reshape(psds,[],n_subjs,n_cond,ne);
% average out the subjects
P_mean=squeeze(mean(P,2));

% subject-averaged models
M_avg=nan(ns,n_cond,ne);
for cc=1:n_cond
    M_avg(:,cc,:)=avg_models(cc).w;
end
X_avg=reshape(M_avg,ns,[]);
[psds_avg,~]=periodogram(X_avg,rectwin(ns),ns,fs);
P_savg=reshape(psds_avg,[],n_cond,ne);


%% plot all individual PSD(TRF) together 

xlims=[0,15];
ylims1=[0 0.02];

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
show_elec=85;
for cc=1:n_cond
    figure
    plot(freqs,squeeze(P_mean(:,cc,show_elec)))
    xlabel('frequency (Hz)')
    title(sprintf('avg(PSD(TRFs)) condition,electrode: %d,%d',cc,show_elec))
    set(gca(),'XLim',xlims,'YLim',ylims1)
end
%% plot PSD(avg(TRFs)) 
ylims2=[0,.01];

% sorted by condition
for cc=1:n_cond
    figure
    plot(freqs,squeeze(P_savg(:,cc,:)))
    xlabel('frequency (Hz)')
    title(sprintf('PSD(avg(TRFs)) - condition: %d',cc))
    set(gca(),'XLim',xlims,'YLim',ylims2)
end
% plot subject-averaged, sorted by condition, for a particular electrode
show_elec=85;
for cc=1:n_cond
    figure
    plot(freqs,squeeze(P_savg(:,cc,show_elec)))
    xlabel('frequency (Hz)')
    title(sprintf('PSD(avg(TRFs)) condition,electrode: %d,%d',cc,show_elec))
    set(gca(),'XLim',xlims,'YLim',ylims2)
end
