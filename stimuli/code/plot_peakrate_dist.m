% plot histogram of peakrate intervals/frequencies for a set of warped
% stimuli

%goals: compare/contrast the anchorpoints vs what our peakrate algorithm
%gives

clear, clc
global boxdir_mine
warp_nm='rule11_seg_bark_median_unnormalizedDurations_varp25';
regularity=1; %-1-> irreg 1-> reg
switch regularity
    case -1
        cond='stretchy_irreg';
    case 1
        cond='compressy_reg';
    otherwise
        error('regularity must be reg or irreg')
end

fs=44100;
max_interval=0.75; % in s
p_thresh=0.105;
w_thresh=2.026;
% p_thresh=0;
% w_thresh=0;


% get warp rule and normalization info from fnm
warp_info=split(warp_nm,'_');
warp_rule=sscanf(warp_info{1},'rule%d');
%% load peakRate data
%TODO: repeat for og? they should match the s_intervals identically but
%could be a good sanity check...
peakRate_dir=sprintf('%s/stimuli/peakRate/',boxdir_mine);
% peakRate_fnm_cond='rule2_seg_bark_median_segmentNormalizedDurations';
peakRate_pth_cnd=fullfile(peakRate_dir,warp_nm);
if isfile([peakRate_pth_cnd '.mat'])
    load(peakRate_pth_cnd);
    warped_peakrate_available=true;
else
    warped_peakrate_available=false;
end

peakRate_fnm_og='og';
peakRate_pth_og=fullfile(peakRate_dir,peakRate_fnm_og);
temp_pr=load(peakRate_pth_og);
peakRate_og=temp_pr.peakRate;


% apply thresholds and stack
alg_intervals_og=[];
alg_intervals_warped=[];
for ss=1:numel(peakRate_og)
    if warped_peakrate_available
        temp_p_mask=peakRate(ss).prominence>p_thresh;
        temp_w_mask=peakRate(ss).peakwidth>w_thresh;
    
        % only care about the times
        temp_times=peakRate(ss).times(temp_w_mask&temp_p_mask);
        temp_intervals=diff(temp_times);
        % alg_intervals_warped=cat(1,alg_intervals_warped, ...
        %     temp_intervals(temp_intervals<=max_interval));
    
        alg_intervals_warped=cat(1,alg_intervals_warped, ...
            temp_intervals);
    end
    
    % rinse and repeat for og - probably bad practice but note recycling temp vars
    
    temp_p_mask=peakRate_og(ss).prominence>p_thresh;
    temp_w_mask=peakRate_og(ss).peakwidth>w_thresh;

    temp_times=peakRate_og(ss).times(temp_w_mask&temp_p_mask);
    temp_intervals=diff(temp_times);
    % alg_intervals_og=cat(1,alg_intervals_og, ...
    %     temp_intervals(temp_intervals<=max_interval));

    alg_intervals_og=cat(1,alg_intervals_og, ...
        temp_intervals);
    
    
    clear temp_p_mask temp_w_mask temp_times temp_intervals



end
alg_intervals_og(alg_intervals_og>max_interval)=[];
if warped_peakrate_available
    alg_intervals_warped(alg_intervals_warped>max_interval)=[];
end
%% load s-mat intervals
% "C:\Users\ninet\Box\my box\LALOR LAB\oscillations project\MATLAB\Warped Speech\stimuli\wrinkle\stretchy_compressy_temp\stretchy_irreg\rule2_seg_bark_median_segment_normalized_durations"
smats_dir=sprintf('%s/stimuli/wrinkle/stretchy_compressy_temp/%s/%s/',boxdir_mine,cond,warp_nm);
D=dir([smats_dir '*.mat']);
% arrange into column vectors
s_intervals_og=[];
s_intervals_warped=[];
for dd=1:numel(D)
    load(fullfile(smats_dir,D(dd).name),'s_temp') 
    % remove start/end anchorpoints
    s_temp([1,end],:)=[];
    s_intervals=diff(s_temp)./fs;
    s_intervals_og=cat(1,s_intervals_og,s_intervals(:,1));
    s_intervals_warped=cat(1,s_intervals_warped,s_intervals(:,2));
    clear s_temp
end
% filter out long pauses
s_intervals_og(s_intervals_og>max_interval)=[];
s_intervals_warped(s_intervals_warped>max_interval)=[];
%% plot s-mat hists

%TODO: fix names so normalization info can also be readily extracted this
%way
ylims=[0 .5]; % make emptyy for no fuks given
hist_config_sw.bin_scale='log';
hist_config_sw.n_bins=50;
hist_config_sw.xlims=[1 34];
hist_config_sw.logTicks=2.^(-1:16);
hist_config_sw.title=sprintf('Anchorpoints - rule%d - regularity: %d',warp_rule,regularity);
hist_config_sw.bin_lims=[.5, 36];

hist_config_sog=hist_config_sw;
hist_config_sog.title=sprintf('Anchorpoints - og');

figure
rates_hist_wrapper(s_intervals_og,hist_config_sog);
if ~isempty(ylims)
    set(gca(),"YLim",ylims)
end

figure
rates_hist_wrapper(s_intervals_warped,hist_config_sw);
if ~isempty(ylims)
    set(gca(),"YLim",ylims)
end

 % plot the actual intervals 
figure
histogram(s_intervals_warped,NumBins=50)
xlabel('time (s)')
title(sprintf('interpeak intervals for rule %d',warp_rule))
xlim([0 1]);

figure
histogram(s_intervals_og)
xlabel('time (s)')
title('og intervals distribution')
xlim([0 1])
%% plot algo hists


ylims=[0 .5]; % make emptyy for no fux given
hist_config_aw.bin_scale='log';
hist_config_aw.n_bins=100;
hist_config_aw.xlims=[1 34];
hist_config_aw.logTicks=2.^(-1:16);
hist_config_aw.title=sprintf('Acoustic Algorithm - rule%d',warp_rule);
hist_config_aw.bin_lims=[.5, 36];


hist_config_aog=hist_config_aw;
hist_config_aog.title=sprintf('Acoustic Algorithm - og');

figure
rates_hist_wrapper(alg_intervals_og,hist_config_aog);
if ~isempty(ylims)
    set(gca(),"YLim",ylims)
end

figure
rates_hist_wrapper(alg_intervals_warped,hist_config_aw);
if ~isempty(ylims)
    set(gca(),"YLim",ylims)
end
%% calculate fano-factors
fano_og_alg=get_fano_factor(alg_intervals_og);
fprintf('og, acoustic algo intervals fano factor: %0.3f\n',fano_og_alg)

fano_warped_smat=get_fano_factor(s_intervals_warped);
fprintf('warped, anchorpoint intervals fano factor: %0.3f\n',fano_warped_smat)

fano_warped_alg=get_fano_factor(alg_intervals_warped);
fprintf('warped, acoustic algo intervals fano factor: %0.3f\n',fano_warped_alg)
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

sim_uniform_rates=true;
if sim_uniform_rates
    sim_config=hist_config_sw;
    sim_config.bin_scale='lin';
    n_sim=numel(s_intervals_warped);
    min_rate=1/.75;
    max_rate=8;
    sim_intervals=1./(min_rate+(max_rate-min_rate).*rand(n_sim,1));
    figure
    rates_hist_wrapper(sim_intervals,sim_config)
    title('simulated uniform rates')
end
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
function fano=get_fano_factor(dist)
    fano=var(dist)/mean(dist);
end

function rates_hist_wrapper(intervals,config)
bin_scale=config.bin_scale;
n_bins=config.n_bins;
bin_lims=config.bin_lims;
tit=config.title;
switch bin_scale
    case 'log'
        freq_bins=logspace(log10(min(bin_lims)),log10(max(bin_lims)),n_bins);% .5-32 Hz ish
    case 'lin'
        freq_bins=linspace(min(bin_lims),max(bin_lims),n_bins);
end
%TODO: detrmine if these vars should be in config
xlims=config.xlims;
logTicks=config.logTicks;

histogram(1./intervals,freq_bins,'Normalization','pdf')
xline(median(1./intervals),'r--','DisplayName',sprintf('median: %0.3f Hz',median(1./intervals)))
xlabel('freq (Hz)')
legend()
title(tit)
set(gca,'Xtick',logTicks,'Xscale','log','XLim',xlims)

end