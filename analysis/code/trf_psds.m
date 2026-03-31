% analyze_trfs
% generate plots of power spectral density 
% using pwelch or periodogram method
% run after plot_trfs to load them first
%% --- SETTINGS ---
psd=struct('param',[],'result',[]);
% average -> psd OR psd -> average
psd.param.method='pwelch';
psd.param.time_range=find(avg_models(1).t>0);
psd.param.fs=ind_models(1).fs;

psd_fig.condition_colors=butterfly_fig.condition_colors;
psd_fig.fz=butterfly_fig.fz;
psd_fig.title_fz=butterfly_fig.title_fz;
psd_fig.leg_lw=butterfly_fig.leg_lw;
psd_fig.lw=butterfly_fig.lw;

%% --- Compute PSDS ---
ns=length(psd.param.time_range);
% individual models
% stack all trfs into 2d mat for efficiency
% stack all subjects into a single matrix M
% M: [time x subject x condition x electrodes]
[n_subjs,n_cond]=size(ind_models);
[~,~,ne]=size(ind_models(1).w); 
if mod(ns,8)~=0
    % only true if we use default
    disp('warning: pwelch will truncate signals when getting psd.')
end

TRFs_4d=nan(ns,n_subjs,n_cond,ne);
for ss=1:n_subjs
    for cc=1:n_cond
        % [time x subject x condition x electrodes]
        TRFs_4d(:,ss,cc,:)=ind_models(ss,cc).w(1,psd.param.time_range,:);
    end
end
% TRFs_2d: [time x TRF waveforms (all subjects, electrodes, conditions)]
TRFS_2d=reshape(TRFs_4d,ns,[]);

[psds_2d,freqs]=get_psd(TRFS_2d,psd);
% note: first dimension should be half the number of samples plus 1
% P: [time x subjects x condition x electrodes]
psds_4d=reshape(psds_2d,[],n_subjs,n_cond,ne);
% average across the subjects
psds_3d_subjavg=squeeze(mean(psds_4d,2));

% subject-averaged models
TRFs_3d_subjavg=nan(ns,n_cond,ne);
for cc=1:n_cond
    TRFs_3d_subjavg(:,cc,:)=avg_models(cc).w(1,psd.param.time_range,:);
end
TRFs_2d_subjavg=reshape(TRFs_3d_subjavg,ns,[]);
[psds_2d_subjavg,~]=get_psd(TRFs_2d_subjavg,psd);
psds_3d_subjavgfirst=reshape(psds_2d_subjavg,[],n_cond,ne);


%% plot all individual PSD(TRF) together 

xlims=[0,15]; % Hz
ylims1=[0 0.015]; %

figure('Name','individual PSD(TRFs) - all subjects all electrodes')
plot(freqs,psds_2d)
grid on
title('UNAVERAGED PSD(TRFs) for all subjects/electrodes',FontSize=psd_fig.title_fz)
xlabel('Frequency (Hz)','FontSize',psd_fig.fz)
set(gca(),'XLim',xlims);
%% plot avg(PSDs(TRFs)) 
% sorted by condition
figure("Name",'avg(PSD(TRFs)',"Color","white")
for cc=1:n_cond
    ax_avglast=subplot(3,2,2*cc-1); hold on;
    plot(freqs,squeeze(psds_3d_subjavg(:,cc,:)), ...
        'Color',psd_fig.condition_colors.(experiment_conditions{cc}))
    grid on
    xlabel('Frequency (Hz)')
    ylabel('PSD? (a.u.)')
    title(sprintf('%s', ...
        experiment_conditions{cc}), 'FontSize',psd_fig.title_fz)
    set(gca(),'XLim',xlims,'YLim',ylims1)

    ax_avglast=subplot(3,2,cc*2); hold on;
    plot(freqs,squeeze(mean(psds_3d_subjavg(:,cc,:),3)), ...
        'Color',psd_fig.condition_colors.(experiment_conditions{cc}))
    grid on
    xlabel('Frequency (Hz)')
    ylabel('PSD? (a.u.)')
    title(sprintf('%s', ...
        experiment_conditions{cc}), 'FontSize',psd_fig.title_fz)
    set(gca(),'XLim',xlims,'YLim',ylims1)
end
sgtitle('avg(PSD(TRFs)) - individual electrodes (left) averaged across electrodes (right)')

%% plot PSD(avg(TRFs)) 
ylims2=[0,.006];
figure('Name','PSD(avg(TRFs))','Color','w')
% sorted by condition
for cc=1:n_cond
    ax_avgfirst=subplot(3,2,cc*2-1);
    plot(freqs,squeeze(psds_3d_subjavgfirst(:,cc,:)), ...
        'Color',psd_fig.condition_colors.(experiment_conditions{cc}))
    grid on
    xlabel('Frequency (Hz)')
    title(experiment_conditions{cc},'FontSize',psd_fig.title_fz)
    set(gca(),'XLim',xlims,'YLim',ylims2)

    ax_avgfirst=subplot(3,2,cc*2);
    plot(freqs,squeeze(mean(psds_3d_subjavgfirst(:,cc,:),3)), ...
        'Color',psd_fig.condition_colors.(experiment_conditions{cc}))
    grid on
    xlabel('Frequency (Hz)')
    title(experiment_conditions{cc},'FontSize',psd_fig.title_fz)
    set(gca(),'XLim',xlims,'YLim',ylims2)
end
sgtitle('PSD(AVG(TRFs) - Left: individual electrodes Right: averaged across electrodes')
% plot subject-averaged, sorted by condition, for a particular electrode
% show_single_elec=true;
% if show_single_elec
%     % show_elec=find(single_pk_electrodes);
%     for cc=1:n_cond
%         figure
%         plot(freqs,squeeze(P_savg(:,cc,single_pk_electrodes)))
%         xlabel('frequency (Hz)')
%         ylabel('PSD (a.u.)')
%         title(sprintf('reliable electrodes - PSD(avg(TRFs)) - condition: %s',conditions{cc}))
%         set(gca(),'XLim',xlims,'YLim',ylims2)
%     end
% end
%% compare PSDs post-onset only against entire timespan ?

%% look at pre-onset PSDs?

%% helpers
function [psd,freqs]=get_psd(X,psd)
% assumes X is 2D matrix that is [time, waveforms] shape
fs=psd.param.fs;
ns=size(X,1);
win_len=round(ns/3);
% win_len=[];
% remove mean
X=detrend(X,'constant');
switch psd.param.method
    case 'periodogram'
        [psd,freqs]=periodogram(X,hamming(ns),ns,fs);
    case 'pwelch'
        %TODO: explore effect of changing window size and noverlap...
        [psd,freqs]=pwelch(X,win_len,[],ns,fs);
end
end