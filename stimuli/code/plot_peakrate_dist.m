% plot histogram of peakrate intervals/frequencies for a set of warped
% stimuli

%goals: compare/contrast the anchorpoints vs what our peakrate algorithm
%gives

clear, clc
global boxdir_mine
cond_nm='rule2_seg_bark_median_segment_normalized_durations';
fs=44100;
max_interval=0.75; % in s
p_thresh=0.105;
w_thresh=2.026;
%% plot peakRate hist
peakRate_dir=sprintf('%s/stimuli/peakRate/',boxdir_mine);
peakRate_fnm='rule2_seg_bark_median_blanket_normalized_durations';
peakRate_pth=fullfile(peakRate_dir,peakRate_fnm);
% load(peakRate_pth);
%% load s-mat intervals
% "C:\Users\ninet\Box\my box\LALOR LAB\oscillations project\MATLAB\Warped Speech\stimuli\wrinkle\stretchy_compressy_temp\stretchy_irreg\rule2_seg_bark_median_segment_normalized_durations"
smats_dir=sprintf('%s/stimuli/wrinkle/stretchy_compressy_temp/stretchy_irreg/%s/',boxdir_mine,cond_nm);
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
%% plot s-mat hist
% get warp rule and normalization info from fnm
warp_info=split(peakRate_fnm,'_');
warp_rule=sscanf(warp_info{1},'rule%d');
%TODO: fix names so normalization info can also be readily extracted this
%way
ylims=[0 .5]; % make emptyy for no fuk given
hist_config_sw.bin_scale='log';
hist_config_sw.n_bins=100;
hist_config_sw.xlims=[1 34];
hist_config_sw.logTicks=2.^(-1:16);
hist_config_sw.title=sprintf('Anchorpoints - rule%d',warp_rule);
hist_config_sw.bin_lims=[.5, 36];

hist_config_og=hist_config_sw;
hist_config_og.title=sprintf('Anchorpoints - og');

figure
hist_wrapper(s_intervals_og,hist_config_og);
if ~isempty(ylims)
    set(gca(),"YLim",ylims)
end

figure
hist_wrapper(s_intervals_warped,hist_config_sw);
if ~isempty(ylims)
    set(gca(),"YLim",ylims)
end
%% calculate fano-factor

% helpers
function hist_wrapper(intervals,config)
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