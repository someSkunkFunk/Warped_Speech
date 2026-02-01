% plot histogram of peakrate intervals/frequencies for a set of warped
% stimuli

%goals: compare/contrast the anchorpoints vs what our peakrate algorithm
%gives

clear, clc
global boxdir_mine
%% set params
% pilot_stimfolder = 'rule11_seg_textgrid_4p545Hz_0_0';
warp_dir='rule14_seg_textgrid_4p545Hz_0_0';
regularity=1; %1-> irreg -1-> reg

[s_intervals,peakrate_mat,warp_config,cond_nm]=load_smat_intervals(regularity,warp_dir);

medians_=median(s_intervals);
og_median=medians_(1);
warped_median=medians_(2);
clear medians_

hist_config=[]; %use defaults
hist_config.normalization='count';
hist_config.domain='time';

condition_colors=struct('original',[1 150 55]./206, ...
    'Regular',[255 0 0]./255, ...
    'Irregular',[0 0 0]);
%% plot og
h=rates_hist_wrapper(s_intervals(:,1),hist_config);
title('og')
set(h,'FaceColor',condition_colors.original)
legend()
%% plot warped condition
if strcmp(cond_nm,'Regular')
    hist_config.n_bins=1;
end
h=rates_hist_wrapper(s_intervals(:,2),hist_config);
title(cond_nm)
legend()
% get warp rule and normalization info from fnm
warp_info=split(warp_dir,'_');
warp_rule=sscanf(warp_info{1},'rule%d');
set(h,'FaceColor',condition_colors.(cond_nm))
%% plot desired reg condition (for proposal)
%% plot warped condition
if strcmp(cond_nm,'Regular')
    hist_config.n_bins=1;
end
ideal_irreg=min(s_intervals(:,2)) + (max(s_intervals(:,2))-min(s_intervals(:,2))).*rand(size(s_intervals(:,2)));
h=rates_hist_wrapper(ideal_irreg,hist_config);
title([cond_nm ' (ideal)'])
legend()
% get warp rule and normalization info from fnm
warp_info=split(warp_dir,'_');
warp_rule=sscanf(warp_info{1},'rule%d');
set(h,'FaceColor',condition_colors.(cond_nm))
%%
fprintf('fano factor for %s: %0.3f\n',cond_nm,get_fano_factor(s_intervals(:,2)))
fprintf('fano factor for og: %0.3f\n',get_fano_factor(s_intervals(:,1)))

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
%note: long pauses should correspond across og/warp
s_intervals(s_intervals(:,1)>=max_interval,:)=[];
% s_intervals_og(s_intervals_og>max_interval)=[];
% s_intervals_warped(s_intervals_warped>max_interval)=[];
% s_intervals=[s_intervals_og,s_intervals_warped];
end

function fano=get_fano_factor(dist)
    fano=var(dist)/mean(dist)^2;
end
function h=rates_hist_wrapper(intervals,hist_config)

freq_defaults=struct('bin_scale','log', ...
    'n_bins',50, ...
    'bin_lims',[.5, 36], ... % .5-32 Hz ish
    'xTicks',2.^(-1:16), ... %log-scale
    'xlims',[1 34],...
    'xScale','log',...
    'normalization','probability', ...
    'domain','freq'...
    );
time_defaults=struct('bin_scale','lin', ...
    'n_bins',50, ...
    'bin_lims',[], ...
    'xTicks',[], ...
    'xlims',[0 1000],... % in ms
    'xScale','lin',...
    'normalization','probability', ...
    'domain','time'...
    );
% copy defaults
if isempty(hist_config.domain)
    error('need to choose rate or time domain for plot.')
else
    switch hist_config.domain
        case 'freq'
            defaults=freq_defaults;
        case 'time'
            defaults=time_defaults;
        otherwise
            error('hist_config.domain=%s not an option',hist_config.domain)
    end

end   
config_fldnms=fieldnames(defaults);
for ff=1:numel(config_fldnms)
    fld=config_fldnms{ff};
    if ~isfield(hist_config,fld)||isempty(hist_config.(fld))
        hist_config.(fld)=defaults.(fld);
    end
end

fz=26; %fontsize
lw=4; % linewidth
bin_scale=hist_config.bin_scale;
n_bins=hist_config.n_bins;
bin_lims=hist_config.bin_lims;
xTicks=hist_config.xTicks;
xlims=hist_config.xlims;
xScale=hist_config.xScale;
normalization=hist_config.normalization;


switch hist_config.domain
    case 'time'
        x=intervals.*1e3;
    case 'freq'
        x=1./intervals;
    otherwise
        error('bruh.')
end

if isempty(bin_lims)
    bin_lims=[min(x),max(x)];
end
switch bin_scale
    case 'log'
        bin_edges=logspace(log10(min(bin_lims)),log10(max(bin_lims)),n_bins);
    case 'lin'
        bin_edges=linspace(min(bin_lims),max(bin_lims),n_bins);
end
if numel(bin_edges)==1
    % avoid histogram bug from ignoring ill-defined edges
    % note the padding amount might only make sense for time
    bin_edges=[min(x)-50 max(x)+0.50];
end

figure('Units','inches','Position',[0 0 6 6])
if isempty(bin_edges)
    h=histogram(x,n_bins,'Normalization',normalization);
else
    h=histogram(x,bin_edges,'Normalization',normalization);
end

switch hist_config.domain
    case 'freq'
        xline(median(x),'r--','LineWidth',lw,'DisplayName',sprintf('median: %0.3g ms',median(x)))
        xlabel('freq (Hz)')
    case 'time'
        xline(median(x),'r--','LineWidth',lw,'DisplayName',sprintf('median: %0.3g ms',median(x)))
        xlabel('inter-syllabic interval (ms)')
    otherwise
        error('bruh.')
end
switch normalization
    case 'count'
        ylabel('Number of syllables')
        ylims=[0,1500];
    otherwise
        ylims=[0,median(x)];
        ylabel(normalization)
end

if isempty(xTicks)
    set(gca,'Xscale',xScale,'XLim',xlims,'FontSize',fz,'YLim',ylims)
else
    set(gca,'Xtick',xTicks,'Xscale',xScale,'XLim',xlims,'FontSize',fz,'YLim',ylims)
end
end