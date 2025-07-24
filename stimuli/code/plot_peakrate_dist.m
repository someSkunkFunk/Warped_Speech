% plot histogram of peakrate intervals/frequencies for a set of warped
% stimuli

%goals: compare/contrast the anchorpoints vs what our peakrate algorithm
%gives

clear, clc
global boxdir_mine
<<<<<<< Updated upstream
cond_nm='rule8_seg_bark_median';
=======
cond_nm='rule8_seg_bark_median_speedNormalizedDurations';
>>>>>>> Stashed changes
fs=44100;
max_interval=0.75; % in s
p_thresh=0.105;
w_thresh=2.026;
% p_thresh=0;
% w_thresh=0;

% get warp rule and normalization info from fnm
warp_info=split(cond_nm,'_');
warp_rule=sscanf(warp_info{1},'rule%d');
%% load peakRate data
%TODO: repeat for og? they should match the s_intervals identically but
%could be a good sanity check...
peakRate_dir=sprintf('%s/stimuli/peakRate/',boxdir_mine);
% peakRate_fnm_cond='rule2_seg_bark_median_segmentNormalizedDurations';
peakRate_pth_cnd=fullfile(peakRate_dir,cond_nm);
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
%% plot s-mat hists

%TODO: fix names so normalization info can also be readily extracted this
%way
ylims=[0 .5]; % make emptyy for no fuks given
hist_config_sw.bin_scale='log';
hist_config_sw.n_bins=100;
hist_config_sw.xlims=[1 34];
hist_config_sw.logTicks=2.^(-1:16);
hist_config_sw.title=sprintf('Anchorpoints - rule%d',warp_rule);
hist_config_sw.bin_lims=[.5, 36];

hist_config_sog=hist_config_sw;
hist_config_sog.title=sprintf('Anchorpoints - og');

figure
hist_wrapper(s_intervals_og,hist_config_sog);
if ~isempty(ylims)
    set(gca(),"YLim",ylims)
end

figure
hist_wrapper(s_intervals_warped,hist_config_sw);
if ~isempty(ylims)
    set(gca(),"YLim",ylims)
end

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
hist_wrapper(alg_intervals_og,hist_config_aog);
if ~isempty(ylims)
    set(gca(),"YLim",ylims)
end

figure
hist_wrapper(alg_intervals_warped,hist_config_aw);
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

%% helpers
function fano=get_fano_factor(dist)
    fano=var(dist)/mean(dist);
end

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