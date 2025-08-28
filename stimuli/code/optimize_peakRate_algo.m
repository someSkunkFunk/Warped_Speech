% script for developing peakRate algorithm bit that filters out overly fast
% events (which can't possibly be syllables)
%TODO: update so baseline plots as subplot on all the thresholded
%histograms, toggle time-domain plot on/off (also inlcude baseline as
%subplot on all thresholded), also add envelope to it for additional visual
%information
%two possibilities
% idea 1:get all the peakRates with peak_tol at 10%, throw out
% lesser peak of every pair that is "too fast"... there may be difficult
% edge cases to consider here 

clear,clc,close all
global boxdir_mine
og_stimuli_dir=sprintf('%s/stimuli/wrinkle_wClicks/og/',boxdir_mine);
stimuli_has_clicks=true;
click_chn=2; %TODO: verify...? from file somehow perhaps....?
stimuli_dir_contents=dir([og_stimuli_dir '*.wav']);
% NOTE: assuming this peakRate file has bark_envelope derived events with
% peak_tol = 10% and no other different parameters from current
% stretchyWrinkle/warp_stimuli_stretchy scripts
peakRate_dir=sprintf('%s/stimuli/peakRate/',boxdir_mine);
baseline_peakRate_file=[peakRate_dir 'og.mat'];
syllable_cutoff_hz=8;
use_cutoff_filter=true; % if true remove peaks that will give stuff above 
% syllable_cutoff_hz... only use if happy with p + w filter results and
% reasonably certain this won't remove real syllables - keeps the first
% peak in such sequences
% NOTE: only using above in the histogram + time domain plots for clarity
clip_duration=64; % in seconds

if exist(baseline_peakRate_file,'file')
    fprintf('loading data from pre-existing %s file\n',baseline_peakRate_file)
    
    temp=load(baseline_peakRate_file,'peakRate');
    peakRate_from_file=true;
    % peakRate: 1x120 struct with fields 'times' 'amplitudes'
    peakRate=temp.peakRate;
    % check if p-vals stored:
    has_pvals=isfield(peakRate,'prominence');
    has_wvals=isfield(peakRate,'peakwidth');
    has_w2vals=isfield(peakRate,'peakwidth2');
    clear temp
else
    fprintf('could not find peakrate data in %s\n',baseline_peakRate_file)
    peakRate_from_file=false;
    has_pvals=false;
    has_wvals=false;
    has_w2vals=false;
end
%% get baseline peakRate (peak_tol = 10 %) and prominence values
%TODO: account for case were baseline file does not exist (if necessary)
%and need to completely re-create it
if ~(has_pvals&&has_wvals&&has_w2vals)
for nn=1:numel(stimuli_dir_contents)

% NOTE: can save time by modifying get_peakRate to accept multiple peak_tol
% vals so we don't have to re-filter it for each value
% TODO: re-create baseline case if not using the existing one (or if it
% does not exists)
fprintf('getting prominence/width vals for stimulus %d of %d\n',nn,numel(stimuli_dir_contents))
wav_path=[og_stimuli_dir stimuli_dir_contents(nn).name];
[wf, wav_fs]=audioread(wav_path);
if stimuli_has_clicks
    wf(:,click_chn)=[];
end
temp_env=bark_env(wf,wav_fs,wav_fs);
% NOTE: original file had prominence normalized by std and width values
% not... updated get_peakRate returns all normalized values

%TODO: is there a way to package peakRate assignment into a functions such
%that
% a. it pre-allocates enough memory if starting from scratch?
% b. it just adds the missing fields/values if loading a pre-existing
% struct from file?
if peakRate_from_file
    % just add threshold vars (and envelope) to existing peakRate struct in
    % file
    [~,~,temp_prominence_vals,temp_width_vals,temp_width2_vals]=get_peakRate(temp_env,wav_fs);
    peakRate(nn).prominence=temp_prominence_vals;
    peakRate(nn).peakwidth=temp_width_vals;
    peakRate(nn).peakwidth2=temp_width2_vals;
    peakRate(nn).env=temp_env;
    clear temp_prominence_vals temp_width_vals temp_width2_vals temp_env
    % sanity check that values make sense with pre-existing peakRate file:
    fprintf('sizes of amplitude/prominence val arrays for current stim:\n')
    disp(size(peakRate(nn).amplitudes))
    disp(size(peakRate(nn).prominence))
else
    % rebuild entire file from scratch
    [temp_times,temp_amps,temp_prominence_vals,temp_width_vals,temp_width2_vals]=get_peakRate(temp_env,wav_fs);
    peakRate(nn).times=temp_times;
    peakRate(nn).amplitudes=temp_amps;
    peakRate(nn).prominence=temp_prominence_vals;
    peakRate(nn).peakwidth=temp_width_vals;
    peakRate(nn).peakwidth2=temp_width2_vals;
    peakRate(nn).env=temp_env;
    clear temp_times temp_amps temp_prominence_vals temp_width_vals temp_width2_vals temp_env
    % sanity check that values make sense with pre-existing peakRate file:
    fprintf('sizes of amplitude/prominence val arrays for current stim:\n')
    disp(size(peakRate(nn).amplitudes))
    disp(size(peakRate(nn).prominence))
end
end
fprintf('saving new p/w vals to %s...\n',baseline_peakRate_file)
if peakRate_from_file
    save(baseline_peakRate_file,"peakRate","-append")
else
    % filter/downsample envelopes before saving
    env_fs=128; %bc that's what we using in trfs
    bark_envs=nan(clip_duration*env_fs,numel(peakRate));
    %same filter as used in get_peakRate
    Hd = getLPFilt(wav_fs,10);
    for nn=1:numel(peakRate)
        % antialias filter
        % NOTE: get_peakrate uses the same filter but maybe makes sense to
        % just give it a pre-filtered envelope (or feed filter as
        % parameter) to avoid having to filter twice
        fprintf('filtering/downsampling envelope %d of %d...\n',nn,numel(peakRate))
        temp_env=filtfilthd(Hd,peakRate(nn).env);
        temp_env=resample(temp_env,env_fs,wav_fs);
        % temp_env=nt_dsample(temp_env,wav_fs/env_fs);
        bark_envs(:,nn)=temp_env;
        clear temp_env
    end
    %remove field from peakRate - note: probably should never be a field to
    %start with but who cares
    peakRate=rmfield(peakRate,'env');
    % technically fs is inferrable from times...
    save(baseline_peakRate_file,"wav_fs","peakRate","bark_envs","env_fs")
    fprintf('new file saved.\n')
end
else
    fprintf('%s \nhas all baseline vals, skipping computation.\n',baseline_peakRate_file)
end
%% what is the range of prominence/width vals?
all_pvals=vertcat(peakRate(:).prominence);
all_wvals=vertcat(peakRate(:).peakwidth);
all_w2vals=vertcat(peakRate(:).peakwidth2);
fprintf('range of prominence vals: [%.02f, %.02f]\n',min(all_pvals),max(all_pvals))
fprintf('range of width vals: [%.02f, %.02f]\n',min(all_wvals),max(all_wvals))
fprintf('range of width2 vals: [%.02f, %.02f]\n',min(all_w2vals),max(all_w2vals))
% don't really need to plot, just need to see behavior of peaks that get
% eliminated as we raise the threshold

%% get baseline distribution
%TODO: figure out why there's a weird bump below 8 Hz now...?
% NOTE: maybe related to median_rates/quantile_rates being yoked to N
% instead of number of thresholds
n_p=20; %number of points for prominence
n_w=20; % number of points for peakwidth
n_thresholds=n_p*n_w;

median_rates=nan(n_thresholds,1); % +1 for baseline...?
quantile_rates=nan(n_thresholds,2);
% note: why did we want these quantiles again?
% re: for smooth transition between distribution extremes
qtls=[.45 .55];
% all_times=vertcat(peakRate(:).times);
[all_times,clip_constants]=get_peak_times(peakRate,clip_duration);
all_rates=calculate_rates(all_times);
% [all_intervals,all_rates,all_times]=get_distributions(peakRate);
fprintf('baseline dist loaded.\n')
% median_rates(1)=median(all_rates);
% quantile_rates(1,:)=quantile(all_rates, qtls);

%% set thresholds
% thresh_opts.use_range='custom';
all_peak_amps=vertcat(peakRate(:).amplitudes);
skip_time_domain_plots=true;
% note: just make one variable equal to zero to view single-variable
% threshold result
max_width=4.0;
max_prom=4.0;
thresh_opts.prominence_range=linspace(0,max_prom,n_p);
% thresh_opts.prominence_range=0;
thresh_opts.width_range=linspace(0,max_width,n_w);
% thresh_opts.width_range=0;
[p_t,w_t]=meshgrid(thresh_opts.prominence_range,thresh_opts.width_range);
% make both 1-D for threshold loop (even though we expand later, it's ok):
p_t=p_t(:)';
w_t=w_t(:)';

n_peaks=numel(all_times);
% broadcast to match number of peaks
P_t=repmat(p_t,n_peaks,1); % [n_peaks X n_thresholds]
W_t=repmat(w_t,n_peaks,1); % [n_peaks X n_thresholds]
% broadcast peak times, prominence, and width for mask
% note: i think broadcasting times matrix is useless since iterating
% through thresholds later anyway?
% T=repmat(all_times,1,n_thresholds); % [n_peaks X n_thresholds]
P=repmat(all_pvals,1,n_thresholds); % [n_peaks X n_thresholds]
thresh_opts.which_width=1;
switch thresh_opts.which_width
    case 1
        % i think this was the better one
        W=repmat(all_wvals,1,n_thresholds);
    case 2
        W=repmat(all_w2vals,1,n_thresholds);
    otherwise
        error('select width 1 or 2 only.')
end
% filter/mask matrix
F=(P>P_t)&(W>W_t);
% preallocate arrays for plots
n_syll_range=nan(n_thresholds,1);
n_too_fast=nan(n_thresholds,1);
if ~skip_time_domain_plots
    [stiched_envs,fs_envs]=load_stitched_envs();
end
%% apply thresholds
ff_rates=apply_thresholds(F,all_times);
for nt=1:n_thresholds
    median_rates(nt)=median(ff_rates{nt});
    quantile_rates(nt,:)=quantile(ff_rates{nt},qtls);
    % report distribution quantiles at different threshold
    sprintf('prominence, width: (%0.3f, %0.3f)\nmedian: %0.3f, quantiles= %0.3f, %0.3f \n', ...
        p_t(nt),w_t(nt),median_rates(nt),quantile_rates(nt,1),quantile_rates(nt,2))
    % also report absolute numbers above/below 8 Hz to see if fast stuff being
    % filtered out at all
    n_too_fast(nt)=sum(ff_rates{nt}>=syllable_cutoff_hz);
    n_syll_range(nt)=sum(ff_rates{nt}<syllable_cutoff_hz);
end
%% plots

%TODO: figure out how to map particular prominence/width vals to single
%index
% plot original distribution
% Set histogram params
%TODO: use plot_config struct to accomplish goal stated below - use
%visSylrateWarpingNew values
% use same scale and bins for direct comparison - be sure to look at low
% frequencies too since raising peakTol might fuck 
hist_param.xlims=[1 34];
hist_param.ylims=[]; % for visual comparison
hist_param.xticks=2.^(-1:16); %log-spaced
% hist_param.hist_buffer=0.5; % seconds to wait after plotting histogram so lines after it show up on top
hist_param.bin_scale='lin';
hist_param.bin_lims=[.5, 36];
hist_param.n_bins=100;
fprintf('histogram params set.\n')

rates_hist_wrapper(all_rates,hist_param)
title('PeakRate without Threshold')

% select a particular distribution/time domain plot to view
% select by index:
% nt_plot=8; 
% select by param vals:
% nt_plot=find(round(p_t,3)==2.526&round(w_t,3)==2.105);

pt_plot=0.105;
wt_plot=2.026;
nt_plot=find(round(p_t,3)==pt_plot&round(w_t,3)==wt_plot);
if isempty(nt_plot)
    % specified point was not grid, just add it
    nt_plot=n_thresholds+1;
    p_t(nt_plot)=pt_plot;
    w_t(nt_plot)=wt_plot;
    F(:,nt_plot)=(all_pvals>pt_plot)&(all_wvals)>wt_plot;
    ff_rates{nt_plot}=calculate_rates(all_times(F(:,nt_plot)));
    
end


if use_cutoff_filter
    % filter by prominence and widths, then remove second peak in any
    % sequence that still gives rate above syllable cutoff
    %TODO: run this process repeteadly/recursively (?) until we have
    %removed all subsequent peaks above cutoff
    ff=F(:,nt_plot);
    % ff_idx=find(ff);
    % ff_rates=calculate_rates(all_times(ff));
    % cutoff_filter_idx=find(ff_rates>syllable_cutoff_hz)+1;
    % % apply hard cutoff:
    % ff(ff_idx(cutoff_filter_idx))=false;
    ff=recursive_cutoff_filter(ff,all_times,syllable_cutoff_hz);
    rates_with_cutoff=calculate_rates(all_times(ff));
    % plot without the hard cutoff for comparison:
    rates_hist_wrapper(calculate_rates(all_times(F(:,nt_plot))),hist_param)
    title(sprintf('JUST Prominence, Width thresholds= %0.3f,%0.3f',p_t(nt_plot),w_t(nt_plot)))
    rates_hist_wrapper(rates_with_cutoff,hist_param)
    title(sprintf('syllable cutoff + Prominence, Width thresholds= %0.3f,%0.3f',p_t(nt_plot),w_t(nt_plot)))
    time_domain_plot_wrapper(all_times,clip_constants,all_peak_amps, ...
            ff,stiched_envs,fs_envs)

    sgtitle(sprintf(['syllable cutoff + Peaks & Envelope before/after %0.3f,%0.3f Prominence,' ...
        ' Width threshold'],p_t(nt_plot),w_t(nt_plot)));
    fprintf('Median with hard cutoff at %d hz (%0.3f, %0.3f p, w): %0.3f\n', ...
        syllable_cutoff_hz,p_t(nt_plot),w_t(nt_plot),median(rates_with_cutoff));
    fprintf('Mean with hard cutoff at %d hz (%0.3f, %0.3f p, w): %0.3f\n', ...
        syllable_cutoff_hz,p_t(nt_plot),w_t(nt_plot),mean(rates_with_cutoff));
    fprintf('Quantiles with hard cutoff at %d hz (%0.3f, %0.3f p, w): %0.3f, %0.3f\n', ...
        syllable_cutoff_hz,p_t(nt_plot),w_t(nt_plot),quantile(rates_with_cutoff,qtls));
    % threshold_plot_wrapper(n_syll_range,n_too_fast,p_t(1:n_thresholds),w_t(1:n_thresholds),thresh_opts,syllable_cutoff_hz)

else
    rates_hist_wrapper(calculate_rates(all_times(F(:,nt_plot))),hist_param)
    title(sprintf('Prominence, Width thresholds= %0.3f,%0.3f',p_t(nt_plot),w_t(nt_plot)))
    
    time_domain_plot_wrapper(all_times,clip_constants,all_peak_amps, ...
            F(:,nt_plot),stiched_envs,fs_envs)
    
    sgtitle(sprintf(['Peaks & Envelope before/after %0.3f,%0.3f Prominence,' ...
        ' Width threshold'],p_t(nt_plot),w_t(nt_plot)));
    
end
threshold_plot_wrapper(n_syll_range,n_too_fast,p_t(1:n_thresholds),w_t(1:n_thresholds),thresh_opts,syllable_cutoff_hz)

median_rates(nt_plot)=median(ff_rates{nt_plot});
quantile_rates(nt_plot,:)=quantile(ff_rates{nt_plot},qtls);

fprintf('Quantiles: %0.3f, %0.3f (prominence,width - %0.3f, %0.3f)\n', ...
    quantile_rates(nt_plot,1),quantile_rates(nt_plot,2),p_t(nt_plot),w_t(nt_plot))
fprintf('median: %0.3f (prominence,width - %0.3f, %0.3f)\n', ...
    median_rates(nt_plot),p_t(nt_plot),w_t(nt_plot))
%% helpers
function ff_cleaned=recursive_cutoff_filter(ff,all_times,syllable_cutoff_hz)
    
    ff_rates=calculate_rates(all_times(ff));
    cutoff_filter_idx=find(ff_rates>syllable_cutoff_hz)+1;
    % apply hard cutoff:
    if isempty(cutoff_filter_idx)
        ff_cleaned=ff;
        return;
    end
    ff_idx=find(ff);
    ff(ff_idx(cutoff_filter_idx))=false;

    % recursive step
    ff_cleaned=recursive_cutoff_filter(ff,all_times,syllable_cutoff_hz);
end
function filtered_rates=apply_thresholds(F,all_times)
% F: [peaks x filters]
n_thresholds=size(F,2);
    filtered_rates=cell(n_thresholds,1);
    for nt=1:n_thresholds
        % M=thresh_var>=thresh_opts.thresh_vals(nt);
        % T=all_times(M);
        fm=F(:,nt);
        filtered_rates{nt}=calculate_rates(all_times(fm));
    end
end
function [envs_stitched,fs]=load_stitched_envs()
    global boxdir_mine
    envs_dir=sprintf('%s/stimuli/wrinkle/',boxdir_mine);
    envs_file=[envs_dir 'WrinkleEnvelopes64hz.mat'];
    envs_data=load(envs_file);
    %TODO: edit file so that it is self-evident which condition corresponds
    %to which cell, by looking at source code found that 2nd cell is og
    fs=envs_data.fs;
    envs=envs_data.env(2,:);
    envs_stitched=vertcat(envs{:});
end
function [all_times,clip_constants]=get_peak_times(peakRate,clip_duration)
% [all_times,clip_constants]=get_peak_times(peakRate,clip_duration)
% NOTE must be original times from peakrate
    all_times=vertcat(peakRate(:).times);
    clip_constants=nan(size(all_times));
    start_clip=1;
    % prev_clips=0;
    for nn=1:numel(peakRate)
        start_time=clip_duration.*(nn-1);
        n_clip_peaks=numel(peakRate(nn).times);
        end_clip=start_clip+n_clip_peaks-1;
        clip_constants(start_clip:end_clip)=start_time;
        start_clip=end_clip+1;
    end
    all_times=all_times+clip_constants;
end


function threshold_plot_wrapper(n_syll_range,n_too_fast,p_t,w_t,thresh_opts,syllable_cutoff_hz)
    % threshold_plot_wrapper(n_syll_range,n_too_fast,p_t,w_t,thresh_opts,syllable_cutoff_hz)
    n_p=numel(thresh_opts.prominence_range);
    n_w=numel(thresh_opts.width_range);
    % n_p=numel(p_t);
    % n_w=numel(w_t);

    % if only 1 param varies, can plot line
    num_vars=1;
    if all(w_t==w_t(1))
        x_t=p_t;
        x_str='prominence';
    elseif all(p_t==p_t(1))
        x_t=w_t;
        x_str='peakwidth';
    else
        num_vars=2;
    end
    switch num_vars
        case 1
            % plot n_syllables in output vs threshold
            ylims=[0 33989];
            figure
            plot(x_t,n_syll_range)
            hold on
            plot(x_t,n_too_fast)
            legend('# rates in syllable range','# rates too fast')
            ylabel('# of rates')
            xlabel([x_str ' threshold'])
            set(gca(),'YLim',ylims)
            title('Single variable threshold')
            hold off
        case 2
            % get unfiltered numbers
            % n_syll_0=n_syll_range(w_t==0&p_t==0);
            % n_fast_0=n_too_fast(w_t==0&p_t==0);
            % normalize numbers to unfiltered number
            f_syll=n_syll_range./(n_syll_range+n_too_fast);
            f_fast=n_too_fast./(n_syll_range+n_too_fast);
            
            % order of dimension sizes has to match what we put in meshgrid
            % originally
            WW_t=reshape(w_t,n_p,n_w);
            PP_t=reshape(p_t,n_p,n_w);
            F_syll=reshape(f_syll,n_p,n_w);
            F_fast=reshape(f_fast,n_p,n_w);
            
            figure
            surf(WW_t,PP_t,F_syll)
            xlabel('Width Threshold')
            ylabel('Prominence Threshold')
            zlabel(sprintf('fraction < %d Hz',syllable_cutoff_hz))

            figure
            surf(WW_t,PP_t,F_fast)
            xlabel('Width Threshold')
            ylabel('Prominence Threshold')
            zlabel(sprintf('fraction > %d Hz',syllable_cutoff_hz))

            % their difference is actually what we want to maximize
            figure
            surf(PP_t,WW_t,F_syll-F_fast)
            xlabel('Prominence threshold')
            ylabel('Width threshold')
            zlabel(sprintf('# rates < %d Hz - # rates >= %d Hz', ...
                syllable_cutoff_hz,syllable_cutoff_hz))

    end
end
function time_domain_plot_wrapper(all_times,clip_constants,all_amplitudes, ...
    thresh_mask,stitched_envs,fs_envs)
% time_domain_plot_wrapper(all_times,all_amplitudes,thresh_mask)
% TODO: stitch all envelopes together for plotting, use fs too for x axis
% next to stemp plot
%TODO: rescale envelopes for vis purposes and determine appropriate ylim
    env_time=(0:1/fs_envs:(numel(stitched_envs)-1)/fs_envs)';
    % xlims=[300 304]; %plot start/end times in seconds
    % xlims=[];
    % ylims=[0 max(all_amplitudes)];
    % normalize peak amplitudes to simplify visualization
    all_amplitudes=normalize(all_amplitudes,'range');
    ylims=[0 1];
    % all_times=all_times+clip_constants;
    thresh_times=all_times(thresh_mask);
    thresh_amps=all_amplitudes(thresh_mask);
    figure
    axs(1)=subplot(2,1,1);
    stem(all_times,all_amplitudes)
    hold on
    plot(env_time,stitched_envs);
    xlabel('time (s)')
    ylabel('amplitude (au)')
    set(gca(),'YLim',ylims)
    axs(2)=subplot(2,1,2);
    fprintf('n peaks in stem plot:%d\n',sum(thresh_mask))
    stem(thresh_times,thresh_amps)
    hold on
    plot(env_time,stitched_envs);
    
    xlabel('time (s)')
    ylabel('amplitude (au)')
    set(gca(),'YLim',ylims)
    % set(gca(),'XLim',xlims,'YLim',ylims)

    linkaxes(axs,'x','y')
    % sgtitle(tit)
end

function rates_hist_wrapper(distribution,hist_param)
% plot distribution of syllable rates
bin_lims=hist_param.bin_lims;
n_bins=hist_param.n_bins;
xticks=hist_param.xticks;
xlims=hist_param.xlims;
ylims=hist_param.ylims;
bin_scale=hist_param.bin_scale;


switch bin_scale
    %frequency bins log or lin space
    case 'log'
        bins=logspace(log10(min(bin_lims)),log10(max(bin_lims)),n_bins);% .5-32 Hz ish
    case 'lin'
        bins=linspace(min(bin_lims),max(bin_lims),n_bins);
end
figure
histogram(distribution,bins,'Normalization','count')
ylabel('counts')
xlabel('syllable rate (Hz)')
set(gca(),'XTick',xticks,'XLim',xlims)
end


function rates=calculate_rates(all_times)
% rates=calculate_rates(all_times)
% all_times: [timex1] 

% note: diff operates along 1st non-singleton dim - assuming that's time in
% case of time x thresholds input
ipis=diff(all_times);
%remove clip edges
if any(ipis<0)
    error('should all be positive now.')
end
% ipis(ipis<0)=[];
rates=1./ipis;

end

