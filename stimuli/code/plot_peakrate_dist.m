% plot histogram of peakrate intervals/frequencies for a set of warped
% stimuli

%goals: compare/contrast the anchorpoints vs what our peakrate algorithm
%gives

clear, clc
global boxdir_mine
warp_dir='rule11_seg_textgrid_4.000Hz_1_0';
regularity=-1; %1-> irreg -1-> reg

[s_intervals,peakrate_mat,warp_config,cond_nm]=load_smat_intervals(regularity,warp_dir);
medians_=median(s_intervals);
og_median=medians_(1);
warped_median=medians_(2);
clear medians_
%%
hist_config=[]; %use defaults
hist_config.normalization='count';
rates_hist_wrapper(s_intervals(:,1),hist_config)
title('og')
xlabel('peakrate (hz)')
ylabel('count')
legend()
clear hist_config
hist_config=[]; %use defaults
hist_config.normalization='count';
rates_hist_wrapper(s_intervals(:,2),hist_config)
ylabel('count')
title(cond_nm)
xlabel('peakrate (hz)')
legend()
%%
% fs=44100;
% max_interval=0.75; % in s
% p_thresh=0.105;
% w_thresh=2.026;


% get warp rule and normalization info from fnm
warp_info=split(warp_dir,'_');
warp_rule=sscanf(warp_info{1},'rule%d');
% %% load precomputed peakRate data
% peakRate_dir=sprintf('%s/stimuli/peakRate/',boxdir_mine);
% % peakRate_fnm_cond='rule2_seg_bark_median_segmentNormalizedDurations';
% peakRate_pth_cnd=fullfile(peakRate_dir,warp_nm);
% if isfile([peakRate_pth_cnd '.mat'])
%     load(peakRate_pth_cnd);
%     warped_peakrate_available=true;
% else
%     warped_peakrate_available=false;
% end
% 
% peakRate_fnm_og='og';
% peakRate_pth_og=fullfile(peakRate_dir,peakRate_fnm_og);
% temp_pr=load(peakRate_pth_og);
% peakRate_og=temp_pr.peakRate;
% 
% 
% % apply thresholds and stack
% alg_intervals_og=[];
% alg_intervals_warped=[];
% for ss=1:numel(peakRate_og)
%     if warped_peakrate_available
%         temp_p_mask=peakRate(ss).prominence>p_thresh;
%         temp_w_mask=peakRate(ss).peakwidth>w_thresh;
% 
%         % only care about the times
%         temp_times=peakRate(ss).times(temp_w_mask&temp_p_mask);
%         temp_intervals=diff(temp_times);
%         % alg_intervals_warped=cat(1,alg_intervals_warped, ...
%         %     temp_intervals(temp_intervals<=max_interval));
% 
%         alg_intervals_warped=cat(1,alg_intervals_warped, ...
%             temp_intervals);
%     end
% 
%     % rinse and repeat for og - probably bad practice but note recycling temp vars
% 
%     temp_p_mask=peakRate_og(ss).prominence>p_thresh;
%     temp_w_mask=peakRate_og(ss).peakwidth>w_thresh;
% 
%     temp_times=peakRate_og(ss).times(temp_w_mask&temp_p_mask);
%     temp_intervals=diff(temp_times);
%     % alg_intervals_og=cat(1,alg_intervals_og, ...
%     %     temp_intervals(temp_intervals<=max_interval));
% 
%     alg_intervals_og=cat(1,alg_intervals_og, ...
%         temp_intervals);
% 
% 
%     clear temp_p_mask temp_w_mask temp_times temp_intervals
% 
% 
% 
% end
% % save a copy including the long pauses
% og_intervals_including_lps=alg_intervals_og;
% alg_intervals_og(alg_intervals_og>max_interval)=[];
% if warped_peakrate_available
%     alg_intervals_warped(alg_intervals_warped>max_interval)=[];
% end
%% load s-mat intervals
% "C:\Users\ninet\Box\my box\LALOR LAB\oscillations project\MATLAB\Warped Speech\stimuli\wrinkle\stretchy_compressy_temp\stretchy_irreg\rule2_seg_bark_median_segment_normalized_durations"
% smats_dir=sprintf('%s/stimuli/wrinkle/stretchy_compressy_temp/%s/%s/',boxdir_mine,cond_dir,warp_nm);
% D=dir([smats_dir '*.mat']);
% % arrange into column vectors
% s_intervals_og=[];
% s_intervals_warped=[];
% for dd=1:numel(D)
%     load(fullfile(smats_dir,D(dd).name),'s_temp') 
%     % remove start/end anchorpoints
%     s_temp([1,end],:)=[];
%     s_intervals=diff(s_temp)./fs;
%     s_intervals_og=cat(1,s_intervals_og,s_intervals(:,1));
%     s_intervals_warped=cat(1,s_intervals_warped,s_intervals(:,2));
%     clear s_temp
% end
% % filter out long pauses
% s_intervals_og(s_intervals_og>max_interval)=[];
% s_intervals_warped(s_intervals_warped>max_interval)=[];
%% plot s-mat hists
% 
% %TODO: fix names so normalization info can also be readily extracted this
% %way
% ylims=[0 .2]; % make emptyy for no fuks given
% hist_config_sw.bin_scale='log';
% hist_config_sw.n_bins=50;
% hist_config_sw.xlims=[1 34];
% hist_config_sw.logTicks=2.^(-1:16);
% % hist_config_sw.title=sprintf('Anchorpoints - rule%d - regularity: %d',warp_rule,regularity);
% hist_config_sw.title=sprintf('%s speech syllable rate distribution',cond_nm);
% hist_config_sw.bin_lims=[.5, 36];
% 
% hist_config_sog=hist_config_sw;
% % hist_config_sog.title=sprintf('Anchorpoints - og');
% hist_config_sog.title='original speech syllable rate distribution';
% 
% figure
% rates_hist_wrapper(s_intervals_og,hist_config_sog);
% if ~isempty(ylims)
%     set(gca(),"YLim",ylims)
% end
% ylabel('probability')
% figure
% rates_hist_wrapper(s_intervals_warped,hist_config_sw);
% ylabel('probability')
% if ~isempty(ylims)
%     set(gca(),"YLim",ylims)
% end
% include_intervals=false;
% if include_intervals
%      % plot the actual intervals 
%     figure
%     histogram(s_intervals_warped,NumBins=50)
%     xlabel('time (s)')
%     title(sprintf('interpeak intervals for rule %d',warp_rule))
%     xlim([0 1]);
% 
%     figure
%     histogram(s_intervals_og)
%     xlabel('time (s)')
%     title('og intervals distribution')
%     xlim([0 1])
% end
%% plot algo hists
% %% calculate fano-factors
% fano_og_alg=get_fano_factor(alg_intervals_og);
% fprintf('og, acoustic algo intervals fano factor: %0.3f\n',fano_og_alg)
% 
% fano_warped_smat=get_fano_factor(s_intervals_warped);
% fprintf('warped, anchorpoint intervals fano factor: %0.3f\n',fano_warped_smat)
% 
% fano_warped_alg=get_fano_factor(alg_intervals_warped);
% fprintf('warped, acoustic algo intervals fano factor: %0.3f\n',fano_warped_alg)
%% Plot truncated Poisson (relevant for rule 8 only)
show_poisson=false;
if show_poisson
    poisson_freqs=linspace(0,16,100);
    poisson_intervals=1./poisson_freqs; % note 1/0 should give Inf... maybe ok?
    % exponential distribution mean ~ should be 1/lambda
    % .2624 is mean of og distribution - 1.65 is corrective factor used to get
    % durations closer to og stimuli
    mu=.2624/1.65;
    exp_max_int=0.75;
    exp_min_int=1/10;
    Fmax_exp=expcdf(exp_max_int,mu);
    Fmin_exp=expcdf(exp_min_int,mu);
    % normalize pdf
    exp_trunc=exppdf(poisson_intervals,mu)./(Fmax_exp-Fmin_exp);
    exp_trunc(poisson_intervals>exp_max_int|poisson_intervals<exp_min_int)=0;
    
    % plot against intervals in our rule 8 output
    
    figure
    plot(poisson_intervals,exp_trunc)
    hold on
    histogram(s_intervals_warped,'Normalization','pdf');
    title('rule 8 interval distribution vs truncated poisson of same params')
    xlabel('seconds')
    
    % xlims([0,1])
    
    % figure
    % plot(poisson_freqs,exp_trunc);
    % xlabel('freq (Hz)')
    % title(sprintf('Truncated Poisson with %0.4f s mean interval',mu))
    % set(gca(),'XTick',hist_config_sw.logTicks,'XLim',hist_config_sw.xlims, ...
    %     'XScale','log')
end
%% simulate a uniform distribution with arbitrary number of
% samples to compare warp results to

%% show that uniform-distributed intervals are no longer uniformly distributed in rates (applies to rule 9)
show_uniform_time=false;
if show_uniform_time
    min_stretch_interval=1/8;
    max_stretch_interval=0.75;
    N=1000;
    rand_intervals=(min_stretch_interval+(max_stretch_interval-min_stretch_interval).*rand(N,1));
    
    % plot intervals
    figure
    histogram(rand_intervals);
    xlabel('time (s)')
    title('intervals dist')
    % plot rates
    figure
    histogram(1./rand_intervals);
    xlabel('freq (Hz)')

end
%% helpers
function [s_intervals,peakrate_mat,warp_config,cond_nm]=load_smat_intervals(regularity,warp_dir)
% s_intervals: 2xintervals matrix
% peakrate_mat: n_peaks-by-4 matrix where cols are pkVals,pkTimes,prom,width
% warp_config: strcut, configuration params that during warp which determine
% peakrate finding algo and intervals in warped stimulus
% cond_nm: str, 'reg' or 'irreg' for plotting titles
global boxdir_mine
switch regularity
    case 1
        cond_dir='stretchy_irreg';
        cond_nm='Irregular';
    case -1
        cond_dir='compressy_reg';
        cond_nm='Regular';
    otherwise
        error('regularity must be reg or irreg')
end
smats_dir=sprintf('%s/stimuli/wrinkle/stretchy_compressy_temp/%s/%s/',boxdir_mine,cond_dir,warp_dir);
D=dir([smats_dir '*.mat']);
% arrange into column vectors
% note: could avoid memory errors here by preallocating
s_intervals=[];
peakrate_mat=[];
for dd=1:numel(D)
    load(fullfile(smats_dir,D(dd).name),'S')
    % fprintf('loading %d/%d...\n',dd,numel(D))
    % all warp_configs should be identical within a directory
    if dd==1
        warp_config=S.warp_config;
        % need fs from audio... todo: add fs to warp_config in S instead...
        [~,fs]=audioread(fullfile(smats_dir,[D(dd).name(1:end-4) '.wav']),[1 1]);
        warp_config.fs=fs;
    end
    s=S.s;
    % remove start/end anchorpoints
    s([1,end],:)=[];
    s_intervals_=diff(s)./fs;
    s_intervals=cat(1,s_intervals,s_intervals_);
    % s_intervals_og=cat(1,s_intervals_og,s_intervals(:,1));
    % s_intervals_warped=cat(1,s_intervals_warped,s_intervals(:,2));
    clear s
    % stack peakRate
    switch warp_config.env_method
        case {'hilbert','bark','gammaChirp','oganian'}
            peakRate=S.peakRate;
            peakrate_mat=cat(1,peakrate_mat, ...
                [peakRate.pkVals,peakRate.pkTimes,peakRate.p,peakRate.w]);

        case 'textgrid'
            syllable_times=S.syllable_times;
            filler_nans=nan(length(syllable_times),1);
            peakrate_mat=cat(1,peakrate_mat,filler_nans,syllable_times, ...
                filler_nans,filler_nans);
    end
    clear peakRate S syllable_times filler_nans
end
% filter out long pauses
max_interval=warp_config.sil_tol;
%note: interval should be unchanged in theory across og/warp
s_intervals(s_intervals(:,1)>max_interval,:)=[];
% s_intervals_og(s_intervals_og>max_interval)=[];
% s_intervals_warped(s_intervals_warped>max_interval)=[];
% s_intervals=[s_intervals_og,s_intervals_warped];
end
%todo: take hist part out of this to run plotting outside of function
% function sim_uniform_rates()
% % for comparison with rule 11 output - should be identical
%     sim_config=hist_config_sw;
%     sim_config.bin_scale='log';
%     %todo: check if accounting for "too fast" makes a difference?
%     n_sim=numel(s_intervals_warped);
%     min_sim_rate=1/.75;
%     max_sim_rate=8;
%     sim_intervals=1./(min_sim_rate+(max_sim_rate-min_sim_rate).*rand(n_sim,1));
%     figure
%     rates_hist_wrapper(sim_intervals,sim_config)
%     title('simulated uniform rates - single sample')
%     ylim([0 0.2])
% 
% 
%     % draw repeated random samples with each having size corresponding to
%     % distinct continuous speech segments to asses distribution distortion
%     long_pauses=find(og_intervals_including_lps>max_interval);
%     % pre-allocate since we know the number of intervals should ultimately
%     % be the same 
%     %NOTE: it's super unclear to me how to filter out the too fast
%     %intervalswhile using the same counting method for intervals between
%     %long pausees but that might be the reason for the distortion... just
%     %gonna plot the resulting distribution when we don't force fast
%     %intervals to remain unchanged, hope it looks uniform, AND that it
%     %still sounds intelligble while the durations remain good to compare
%     %with reg
%     sim_intervals_lps=nan(n_sim,1);
%     too_fast=(1./og_intervals_including_lps)>max_sim_rate;
%     sim_intervals_lps(too_fast)=og_intervals_including_lps(too_fast);
%     curr_seg_start=1;
%     for ss=1:numel(long_pauses)
%         if ss<numel(long_pauses)
%             % get number of intervals between long pauses
%             n_sim_seg=long_pauses(ss+1)-long_pauses(ss)-1;
%             if ss==1 && long_pauses(ss)~=1
%                 n_sim_seg=long_pauses(ss)-1+n_sim_seg;
%             end
%         else
%             % get number of intervals between final long pause and
%             % remaining intervals
%             n_sim_seg=numel(og_intervals_including_lps)-long_pauses(ss);
%         end
%         sim_intervals_lps(curr_seg_start:n_sim_seg+curr_seg_start-1)=1./(min_sim_rate+(max_sim_rate-min_sim_rate).*rand(n_sim_seg,1));
%         curr_seg_start=n_sim_seg+curr_seg_start;
%     end
%     figure
%     rates_hist_wrapper(sim_intervals_lps,sim_config)
%     title('simulated uniform rates - multi sample')
%     ylim([0 0.15])
% end
function fano=get_fano_factor(dist)
    fano=var(dist)/mean(dist);
end
function rates_hist_wrapper(intervals,hist_config)
%todo: normalization
defaults=struct('bin_scale','log', ...
    'n_bins',50, ...
    'bin_lims',[.5, 36], ...
    'logTicks',2.^(-1:16), ...
    'xlims',[1 34],...
    'normalization','probability' ...
    );
% copy defaults
config_fldnms=fieldnames(defaults);
for ff=1:numel(config_fldnms)
    fld=config_fldnms{ff};
    if ~isfield(hist_config,fld)||isempty(hist_config.(fld))
        hist_config.(fld)=defaults.(fld);
    end
end
bin_scale=hist_config.bin_scale;
n_bins=hist_config.n_bins;
bin_lims=hist_config.bin_lims;
logTicks=hist_config.logTicks;
xlims=hist_config.xlims;
normalization=hist_config.normalization;

switch bin_scale
    case 'log'
        freq_bins=logspace(log10(min(bin_lims)),log10(max(bin_lims)),n_bins);% .5-32 Hz ish
    case 'lin'
        freq_bins=linspace(min(bin_lims),max(bin_lims),n_bins);
end
figure
histogram(1./intervals,freq_bins,'Normalization',normalization)
xline(median(1./intervals),'r--','DisplayName',sprintf('median: %0.3f Hz',median(1./intervals)))
xlabel('freq (Hz)')
set(gca,'Xtick',logTicks,'Xscale','log','XLim',xlims)

end