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

stim_info.clip_duration=64; % in seconds

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
    bark_envs=nan(stim_info.clip_duration*env_fs,numel(peakRate));
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
%% Set histogram params
%TODO: use plot_config struct to accomplish goal stated below - use
%visSylrateWarpingNew values
% use same scale and bins for direct comparison - be sure to look at low
% frequencies too since raising peakTol might fuck 
hist_param.xlims=[1 34];
hist_param.ylims=[0,.25]; % for visual comparison
hist_param.xticks=2.^(-1:16); %log-spaced
% hist_param.hist_buffer=0.5; % seconds to wait after plotting histogram so lines after it show up on top
hist_param.bin_scale='lin';
hist_param.bin_lims=[.5, 36];
hist_param.n_bins=100;
fprintf('histogram params set.\n')
%% get baseline distribution
n_thresholds=4;
median_rates=nan(n_thresholds+1,1);
quantile_rates=nan(n_thresholds+1,2);
qtls=[.45 .55];
% all_times=vertcat(peakRate(:).times);
[all_times,clip_constants]=get_peak_times(peakRate,stim_info);
all_rates=calculate_rates(all_times);
% [all_intervals,all_rates,all_times]=get_distributions(peakRate);
fprintf('baseline dist loaded.\n')
%% plot baseline distribution

median_rates(1)=median(all_rates);
quantile_rates(1,:)=quantile(all_rates, qtls);
%TODO: remove this once hist_wrapper is updated to include it in subplots
%of each thresholded hist
hist_wrapper(all_rates,'baseline',hist_param)
%% Find median and for particular threshold
all_peak_amps=vertcat(peakRate(:).amplitudes);
skip_time_domain_plot=false;

thresh_info.which_threshold='p';
switch thresh_info.which_threshold
    case 'w'
        thresh_var=all_wvals;
        thresh_info.tit_thresh='peakwidth';
    case 'w2'
        thresh_var=all_w2vals;
        thresh_info.tit_thresh='peakwidth2';
    case 'p'
        thresh_var=all_pvals;
        thresh_info.tit_thresh='prominence';
    case {'w*p','p*w'}
        all_wpvals=all_wvals.*all_pvals;
        thresh_var=all_wpvals;
        thresh_info.tit_thresh='peakwidth*prominence';
    case {'w2*p','p*w2'}
        all_w2pvals=all_w2vals.*all_pvals;
        thresh_var=all_w2pvals;
        thresh_info.tit_thresh='peakwidth2*prominence';
end
thresh_info.thresh_vals=linspace(min(thresh_var),max(thresh_var)/2,n_thresholds);
n_syll_range=nan(n_thresholds,1);
n_too_fast=nan(n_thresholds,1);
if ~skip_time_domain_plot
    [stiched_envs,fs_envs]=load_stitched_envs();
end

for nt=1:n_thresholds
    %TODO: fix thresh_mask error
    thresh_mask=thresh_var>=thresh_info.thresh_vals(nt);
    
    
    temp_thresh_rates=calculate_rates(all_times(thresh_mask));
    
    median_rates(1+nt)=median(temp_thresh_rates);
    quantile_rates(1+nt,:)=quantile(temp_thresh_rates,qtls);
    temp_tit=sprintf('%s threshold=%0.3f',thresh_info.tit_thresh,thresh_info.thresh_vals(nt));
    % report distribution quantiles at different threshold
    fprintf('%0.3f %s_thresh - median=%0.3f, quantiles= %0.3f, %0.3f \n', ...
        thresh_info.thresh_vals(nt),thresh_info.tit_thresh,median_rates(1+nt),quantile_rates(1+nt,1),quantile_rates(1+nt,2))
    % also report absolute numbers above/below 8 Hz to see if fast stuff being
    % filtered out at all
    n_too_fast(nt)=sum(temp_thresh_rates>=8);
    n_syll_range(nt)=sum(temp_thresh_rates<8);
    fprintf('number of peaks above 8 Hz: %d\n', n_too_fast(nt))
    fprintf('number of peaks below 8 Hz: %d\n', n_syll_range(nt))
    fprintf('their sum: %d, \ntotal sum of thresh mask:%d\n',sum([n_too_fast(nt) n_syll_range(nt)]),sum(thresh_mask))
    fprintf('their diff: %d',sum([n_too_fast(nt) n_syll_range(nt)])-sum(thresh_mask))
    
    hist_wrapper(temp_thresh_rates,temp_tit,hist_param)
    if ~skip_time_domain_plot
        time_domain_plot_wrapper(all_times,clip_constants,all_peak_amps, ...
            thresh_mask,temp_tit,stiched_envs,fs_envs)
    end
    % time_domain_plot_wrapper(temp_times,temp_amps,temp_tit)
    clear temp_rates temp_times temp_amps It
end

threshold_distribution_plot_wrapper(n_syll_range,n_too_fast,thresh_info)

%% helpers
function [envs_stitched,fs]=load_stitched_envs()
    global boxdir_mine
    envs_dir=sprintf('%s/stimuli/',boxdir_mine);
    envs_file=[envs_dir 'WrinkleEnvelopes64hz.mat'];
    envs_data=load(envs_file);
    %TODO: edit file so that it is self-evident which condition corresponds
    %to which cell, by looking at source code found that 2nd cell is og
    fs=envs_data.fs;
    envs=envs_data.env(2,:);
    envs_stitched=vertcat(envs{:});
end
function [all_times,clip_constants]=get_peak_times(peakRate,stim_info)
% NOTE must be original times from peakrate
% NOTE: i think the calculate rate will NOT work if we add the clip
% constants so leaving them separate for now
    all_times=vertcat(peakRate(:).times);
    clip_constants=nan(size(all_times));
    start_clip=1;
    % prev_clips=0;
    for nn=1:numel(peakRate)
        start_time=stim_info.clip_duration.*(nn-1);
        n_clip_peaks=numel(peakRate(nn).times);
        end_clip=start_clip+n_clip_peaks-1;
        clip_constants(start_clip:end_clip)=start_time;
        start_clip=end_clip+1;
    end
end
function threshold_distribution_plot_wrapper(n_syll_range,n_too_fast,thresh_info)
    % plot n_syllables in output vs threshold
    ylims=[0 33989];
    figure
    plot(thresh_info.thresh_vals,n_syll_range)
    hold on
    plot(thresh_info.thresh_vals,n_too_fast)
    legend('# rates in syllable range','# rates too fast')
    ylabel('# of rates')
    xlabel('threshold')
    set(gca(),'YLim',ylims)
    title(sprintf('%s',thresh_info.tit_thresh))
    hold off
end
function time_domain_plot_wrapper(all_times,clip_constants,all_amplitudes, ...
    thresh_mask,tit_thresh,stitched_envs,fs_envs)
% time_domain_plot_wrapper(all_times,all_amplitudes,thresh_mask)
% TODO: stitch all envelopes together for plotting, use fs too for x axis
% next to stemp plot
%TODO: rescale envelopes for vis purposes and determine appropriate ylim
    tit=sprintf('Peaks & Envelope before/after %s threshold',tit_thresh);
    env_time=(0:1/fs_envs:(numel(stitched_envs)-1)/fs_envs)';
    % xlims=[300 304]; %plot start/end times in seconds
    % xlims=[];
    % ylims=[0 max(all_amplitudes)];
    % normalize peak amplitudes to simplify visualization
    all_amplitudes=normalize(all_amplitudes,'range');
    ylims=[0 1];
    all_times_stitched=all_times+clip_constants;
    thresh_times=all_times_stitched(thresh_mask);
    thresh_amps=all_amplitudes(thresh_mask);
    figure
    axs(1)=subplot(2,1,1);
    stem(all_times_stitched,all_amplitudes)
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
    sgtitle(tit)
end

function hist_wrapper(distribution,tit,hist_param)
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
histogram(distribution,bins,'Normalization','pdf')
ylabel('probability ')
xlabel('syllable rate (Hz)')
set(gca(),'XTick',xticks,'XLim',xlims,'YLim',ylims)
title(tit)
end


function rates=calculate_rates(all_times,thresh_mask)

if nargin<2||isempty(thresh_mask)
% keep all values provided by default
    thresh_mask=ones(size(all_times,1),1,'logical');
end
if size(all_times)~=size(thresh_mask)
    error('this shit wont work')
end

ipis=diff(all_times(thresh_mask));
%remove clip edges
ipis(ipis<0)=[];
rates=1./ipis;

% I thought this was smart then realized clip_constants are not needed at
% all since clip edges are only intervals that will come out negative and
% can just prune them
    % arguments
    %     all_times
    %     clip_constants
    %     thresh_mask
    % end
    % if size(all_times,1)~=size(clip_constants,1)
    %     error('this shit wont work')
    % end
    % npeaks=size(all_times,1);
    % 
    % if nargin <3||isempty(thresh_mask)
    %     % keep all values provided by default
    %     thresh_mask=ones(npeaks,1);
    % end
    % 
    % clip_start_mask=min(diff(clip_constants),1);
    % % assuming clip constants only increase, this won't work otherwise
    % if any(clip_start_mask<0)
    %     error('this shit wont work')
    % end
    % ipis=diff()
    
end

%% REALM OF BANISHMENT
function [intervals,rates,all_times]=get_distributions(peakRate,config)
% [intervals, rates, peak_vals,all_times]=get_distributions(peakRate)
% from absolute time (all_times) -> interpeak intervals/rates
    % add 1-min correction to each clip
    
    all_times=vertcat(peakRate(:).times);
    % clip_constants=nan(size(all_times));
    % start_clip=1;
    % % prev_clips=0;
    % for nn=1:numel(peakRate)
    %     start_time=config.clip_duration.*(nn-1);
    %     n_clip_peaks=numel(peakRate(nn).times);
    %     end_clip=start_clip+n_clip_peaks-1;
    %     clip_constants(start_clip:end_clip)=start_time;
    %     start_clip=end_clip+1;
    % end
    % all_times=all_times+clip_constants;
    % intervals=diff(all_times);
    % rates=1./intervals;
end