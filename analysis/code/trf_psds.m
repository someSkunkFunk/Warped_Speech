% analyze_trfs
% generate plots of power spectral density 
% using pwelch or periodogram method
% run after plot_trfs to load them first
%% --- SETTINGS ---
psd=struct('param',[],'result',[]);
% average -> psd OR psd -> average
psd.param.method='pwelch';
psd.param.time_range_idx=find(avg_models(1).t>0&avg_models(1).t<=400); 
psd.param.time_range=[avg_models(1).t(psd.param.time_range_idx(1)), ...
                        avg_models(1).t(psd.param.time_range_idx(end))];
psd.param.fs=ind_models(1).fs;
psd.param.nfft=length(psd.param.time_range_idx);
% win_length -> frequency resolution (need long window to get good
% resolution at low frequencies)
% freq res = fs/N_window
psd.param.win_len=psd.param.nfft;
psd.param.noverlap=[]; %NOTE: default is 50% but if win_length=nfft, there is only one window? so no overlap?

psd_fig.condition_colors=butterfly_fig.condition_colors;
psd_fig.fz=butterfly_fig.fz;
psd_fig.title_fz=butterfly_fig.title_fz;
psd_fig.leg_lw=butterfly_fig.leg_lw;
psd_fig.lw=butterfly_fig.lw;
psd_fig.freqlims=[0,15]; % Hz
psd_fig.ylims_avglast=[0 0.033];
psd_fig.ylims_avgfirst=[0 .012];


%% --- Compute PSDS ---

% individual models
% stack all trfs into 2d mat for efficiency
% stack all subjects into a single matrix M
% M: [time x subject x condition x electrodes]
[n_subjs,n_cond]=size(ind_models);
ns=length(psd.param.time_range_idx);
[~,~,ne]=size(ind_models(1).w); 
if mod(ns,8)~=0
    % only true if we use default number of segements
    disp('warning: pwelch will truncate signals when getting psd.')
end

TRFs_4d=nan(ns,n_subjs,n_cond,ne);
for ss=1:n_subjs
    for cc=1:n_cond
        % [time x subject x condition x electrodes]
        TRFs_4d(:,ss,cc,:)=ind_models(ss,cc).w(1,psd.param.time_range_idx,:);
    end
end
% TRFs_2d: [time x TRF waveforms (subjects*electrodes*conditions)]
TRFS_2d=reshape(TRFs_4d,ns,[]);
% psds_2d: [freqs, subjects*electrodes*conditions]
[psds_2d_avglast,freqs]=get_psd(TRFS_2d,psd);
% note: first dimension should be half the number of samples plus 1
% psds_4d: [freq x subjects x condition x electrodes]
psds_4d=reshape(psds_2d_avglast,[],n_subjs,n_cond,ne);
% average across the subjects
% [freq x condition x electrodes]
psds_3d_subjavglast=squeeze(mean(psds_4d,2));

% subject-averaged models
% [time x conditions x electrodes]
TRFs_3d_subjavgfirst=nan(ns,n_cond,ne);
for cc=1:n_cond
    TRFs_3d_subjavgfirst(:,cc,:)=avg_models(cc).w(1,psd.param.time_range_idx,:);
end
% [time x conditions*electrodes]
TRFs_2d_subjavgfirst=reshape(TRFs_3d_subjavgfirst,ns,[]);
% psds_2d_subjav: [freqs x conditions*electrodes]
[psds_2d_subjavgfirst,~]=get_psd(TRFs_2d_subjavgfirst,psd);
% back to [freqs x conditions x electrodes]
psds_3d_subjavgfirst=reshape(psds_2d_subjavgfirst,[],n_cond,ne);


%% plot all individual PSD(TRF) together 


figure('Name','individual PSD(TRFs) - all subjects all electrodes')
plot(freqs,psds_2d_avglast)
grid on
title('UNAVERAGED PSD(TRFs) for all subjects/electrodes',FontSize=psd_fig.title_fz)
xlabel('Frequency (Hz)','FontSize',psd_fig.fz)
set(gca(),'XLim',psd_fig.freqlims);
%% plot avg(PSDs(TRFs)) 
% sorted by condition
figure("Name",'avg(PSD(TRFs)',"Color","white")
for cc=1:n_cond
    ax_avglast=subplot(3,2,2*cc-1); hold on;
    plot(freqs,squeeze(psds_3d_subjavglast(:,cc,:)), ...
        'Color',psd_fig.condition_colors.(experiment_conditions{cc}))
    grid on
    xlabel('Frequency (Hz)')
    ylabel('PSD? (a.u.)')
    title(sprintf('%s', ...
        experiment_conditions{cc}), 'FontSize',psd_fig.title_fz)
    set(gca(),'XLim',psd_fig.freqlims,'YLim',psd_fig.ylims_avglast)

    ax_avglast=subplot(3,2,cc*2); hold on;
    plot(freqs,squeeze(mean(psds_3d_subjavglast(:,cc,:),3)), ...
        'Color',psd_fig.condition_colors.(experiment_conditions{cc}))
    grid on
    xlabel('Frequency (Hz)')
    ylabel('PSD? (a.u.)')
    title(sprintf('%s', ...
        experiment_conditions{cc}), 'FontSize',psd_fig.title_fz)
    set(gca(),'XLim',psd_fig.freqlims,'YLim',psd_fig.ylims_avglast)
end
sgtitle('avg(PSD(TRFs)) - individual chns (left) avg across chns (right)')

%% plot PSD(avg(TRFs)) 

figure('Name','PSD(avg(TRFs))','Color','w')
% sorted by condition
for cc=1:n_cond
    ax_avgfirst=subplot(3,2,cc*2-1);
    plot(freqs,squeeze(psds_3d_subjavgfirst(:,cc,:)), ...
        'Color',psd_fig.condition_colors.(experiment_conditions{cc}))
    grid on
    xlabel('Frequency (Hz)')
    title(experiment_conditions{cc},'FontSize',psd_fig.title_fz)
    set(gca(),'XLim',psd_fig.freqlims,'YLim',psd_fig.ylims_avgfirst)

    ax_avgfirst=subplot(3,2,cc*2);
    plot(freqs,squeeze(mean(psds_3d_subjavgfirst(:,cc,:),3)), ...
        'Color',psd_fig.condition_colors.(experiment_conditions{cc}))
    grid on
    xlabel('Frequency (Hz)')
    title(experiment_conditions{cc},'FontSize',psd_fig.title_fz)
    set(gca(),'XLim',psd_fig.freqlims,'YLim',psd_fig.ylims_avgfirst)
end
sgtitle('PSD(AVG(TRFs) - individual chns (left) avg across chns (right)')
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
%         set(gca(),'XLim',xlims,'YLim',psd_fig.ylims_avgfirst)
%     end
% end
%% compare PSDs post-onset only against entire timespan ?

%% look at pre-onset PSDs?

%% helpers
function [psd,freqs]=get_psd(X,psd)
% assumes X is 2D matrix that is [time, waveforms] shape
fs=psd.param.fs;
nfft=psd.param.nfft;
win_len=psd.param.win_len;
% win_len=[];
% remove mean
X=detrend(X,'constant');
switch psd.param.method
    case 'periodogram'
        [psd,freqs]=periodogram(X,hamming(nfft),nfft,fs);
    case 'pwelch'
        %TODO: explore effect of changing window size and noverlap...
        [psd,freqs]=pwelch(X,win_len,[],nfft,fs);
end
end