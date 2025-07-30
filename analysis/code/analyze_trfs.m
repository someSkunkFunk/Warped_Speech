% analyze_trfs
% run after plot_trfs to load them first
clearvars -except ind_models avg_models trf_config subjs 
clc
%% Compute PSDS
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

% preallocate mem
% columns should be time vectors
% X=nan(ns,n_subjs*n_cond*ne);
% psds=nan(floor(ns/2)+1,n_subjs*n_cond*ne);
% % assuming reshape preserves order of linear indices, we should be able to
% % undo this easily:
% for sc=1:numel(ind_models)
%     X(:,=squeeze(ind_models(sc).w);
% end
% X=squeeze(ind_models(1).w);
X=reshape(M,ns,[]);
[psds,freqs]=periodogram(X,rectwin(ns),ns,fs);

%%
% plot them all together 
xlims=[0,15];
figure
plot(freqs,psds)
title('trf psds for all subjects/electrodes')
xlabel('frequency (Hz)')
xlim(xlims)
% note: first dimension should be half the number of samples plus 1
P=reshape(psds,[],n_subjs,n_cond,ne);
% average out the subjects

% plot subject-averaged, sorted by condition
P_mean=squeeze(mean(P,2));
for cc=1:n_cond
    figure
    plot(freqs,squeeze(P_mean(:,cc,:)))
    xlabel('frequency (Hz)')
    title(sprintf('PSD(subject-averaged TRFs) - condition: %d',cc))
    xlim(xlims)
end


% plot subject-averaged, sorted by condition, for a particular electrode
show_elec=85;
for cc=1:n_cond
    figure
    plot(freqs,squeeze(P_mean(:,cc,show_elec)))
    xlabel('frequency (Hz)')
    title(sprintf('PSD(subject-averaged TRFs) condition,electrode: %d,%d',cc,show_elec))
    xlim(xlims)
end
