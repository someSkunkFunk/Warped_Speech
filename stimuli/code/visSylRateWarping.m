
clear, clc
warpedStimuliDir="./median_stretchy_rule1_seg/wrinkle";
conditions={'reg_stretchy','irreg_stretchy'};

%% read in S from warped stimuli and get peakRate distribution from it
% for cc=1:length(conditions)
%     cond=conditions{cc};
%     condDir=sprintf('%s/%s',warpedStimuliDir,cond);
%     Ds=dir([condDir '/*.mat']);
%     npeaks_previous=0;
%     for dd=1:length(Ds)
%         stimFileName=sprintf('%s/%s',condDir,Ds(dd).name);
%         % read in fs so i dont feel bad about hard coding it even though it
%         % legit does not matter since we've never once changed it 
%         if dd==length(Ds) && cc==length(conditions)
%             [~,fs]=audioread([stimFileName(1:end-4) '.wav']);
%         end
%         load(stimFileName,'s_temp');
%         npeaks_current=size(s_temp,1)-2; %ignore start/end anchorpoints
%         % og intervals only need to be computed once since should be the
%         % same between the two conditions
%         pkIdx=(1:npeaks_current-1)+npeaks_previous;
%         if cc==1
%             ogPeakIntervals(pkIdx,1)=diff(s_temp(2:end-1,1));
%         end
%         warpedPeakIntervals(pkIdx,cc)=diff(s_temp(2:end-1,2));
%         npeaks_previous=pkIdx(end);
%     end
% end
% % convert back to time and get frequencies too
% ogPeakIntervals=ogPeakIntervals./fs;
% warpedPeakIntervals=warpedPeakIntervals./fs;
% ogPeakFreqs=1./ogPeakIntervals;
% warpedPeakFreqs=1./warpedPeakIntervals;
%% SET TOY DIST PLOT PARAMS
% choose a symmetric distribution to test warp rules on
% TODO: allow plotting all rule/distribution combinations via looping
distribution='beta'; % 'beta' or 'uniform' or 'data'
warp_rule=1;

f_center=5.9276; % median of gamma including long pauses
hist_buffer=0.5; % seconds to wait after plotting histogram so lines after it show up on top
bin_scale='lin';
bin_lims=[.5, 32];
n_bins=100;
switch bin_scale
    case 'log'
        % freq_bins=logspace(-1,1.2,100);% .5-16 Hz ish
        freq_bins=logspace(log10(min(bin_lims)),log10(max(bin_lims)),n_bins);% .5-32 Hz ish
    case 'lin'
        freq_bins=linspace(min(bin_lims),max(bin_lims),n_bins);
end
xlims=[1 34];
logTicks=2.^(-1:16);
%% plot warp rule + real dist
figure
subplot(4,1,1)
hold on
plot_stretchy_compressy_rule1(f_center)
% NOTE WE DONT ACTUALLY HAVE STIMULI FROM USING RULE 1 YET SO HISTOGRAMS
% SHOW WARPING WITHOUT THE DISTANCE FROM CENTER DISCOUNT
hold off
subplot(4,1,2)
hold on

%TODO: add a scatter of distribution using color coding to see which points
%crossed median line in reg case
og_mean=mean(ogPeakFreqs);
histogram(ogPeakFreqs,freq_bins,'DisplayName',sprintf('Original syllable rates %s bins', bin_scale));
pause(hist_buffer)
xline(f_center,'k--','DisplayName','og Median')
% xline(og_mean,'b--','DisplayName','og mean')

% plot([f_center, f_center],[0 median_line_max],'--','DisplayName','Median')
hold off
xlabel('freq (hz)')
ylabel('counts')
title('original syllable rate distribution')
set(gca,'Xtick',logTicks,'Xscale','log','XLim',xlims)
legend()
for cc=1:numel(conditions)
    subplot(4,1,2+cc)
    hold on
    % filtIndx=warpedPeakFreqs(:,cc)>2;
    % filt_median=median(warpedPeakFreqs(filtIndx,cc));
    new_median=median(warpedPeakFreqs(:,cc));
    % new_mean=mean(warpedPeakFreqs(:,cc));
    histogram(warpedPeakFreqs(:,cc),freq_bins,'DisplayName', ...
        sprintf('%s syllable rates',conditions{cc}));
    pause(hist_buffer)
    xline(f_center,'k--','DisplayName','OG Median')
    % plot([f_center, f_center],[0 median_line_max],'--','DisplayName','Median')
    xline(new_median,'r--','DisplayName','new median')
    % xline(new_mean,'r--','DisplayName','new_mean')
    % xline(filt_median,'--','DisplayName','Median above 2Hz')
    hold off
    xlabel('freq (hz)')
    ylabel('counts')
    title('Warped syllable rate distribution')
    set(gca,'Xtick',logTicks,'Xscale','log','XLim',xlims)
    legend()
    hold off
end
sgtitle('Actual shit')
%% plot simulated before/after distributions from beta
n_samps=50000;
% sim_shift=0.4;
% inc_factor=(1+sim_shift);
% dec_factor=1./inc_factor;
% sim_min=2;
% sim_max=16;

% plot symmetric beta distribution before/after
beta_a=4;
beta_b=4;

beta_og=betarnd(beta_a,beta_b,n_samps,1);
%rescale to 2-16 hz range
beta_og=rescale_nums(beta_og);
% beta_og=(sim_max-sim_min).*beta_og+sim_min;
median_beta_og=median(beta_og);
mean_beta_og=mean(beta_og);
% RULE 1
[beta_stretchy,beta_compressy]=stretchy_compressy_rule1(beta_og);
% get new medians
median_beta_stretchy=median(beta_stretchy);
median_beta_compressy=median(beta_compressy);
beta_bins_scale='log';
switch beta_bins_scale
    case 'log'
        % .1 - 22 Hzish
        beta_bins=logspace(-1,1.35,100);
    case 'lin'
        beta_bins=linspace(.1,22,100);
end
figure
subplot(4,1,1)
hold on
plot_stretchy_compressy_rule1(median_beta_og)
hold off
subplot(4,1,2)
hold on
histogram(beta_og,beta_bins,'Normalization','pdf','DisplayName','Symmetric beta')
pause(hist_buffer)
xline(median_beta_og,'k--','DisplayName','og median')
xlabel('freq (Hz)')
legend()
title('Parent distribution')
set(gca,'XLim',xlims)
hold off
subplot(4,1,3)
hold on
histogram(beta_compressy,beta_bins,'Normalization','pdf','DisplayName','compressy beta')
pause(hist_buffer)
xline(median_beta_og,'k--','DisplayName',sprintf('og median - %0.2f',median_beta_og))
xline(median_beta_compressy,'r--','DisplayName',sprintf('new median - %0.f',median_beta_compressy))
xlabel('freq (Hz)')
legend()
title('compressy-reg distribution')
set(gca,'XLim',xlims)
hold off
subplot(4,1,4)
hold on
histogram(beta_stretchy,beta_bins,'Normalization','pdf','DisplayName','stretchy beta')
pause(hist_buffer)
xline(median_beta_og,'k--','DisplayName',sprintf('og median - %0.2f',median_beta_og))
xline(median_beta_stretchy,'r--','DisplayName',sprintf('new median - %0.2f',median_beta_stretchy))
xlabel('freq (Hz)')
legend()
title('stretchy-irreg distribution')
set(gca,'XLim',xlims)
hold off
sgtitle('Rule 1 - Symmetric Beta parent distribution')

% RULE 2
[beta_stretchy,beta_compressy]=stretchy_compressy_rule2(beta_og);
% get new medians
median_beta_stretchy=median(beta_stretchy);
median_beta_compressy=median(beta_compressy);

figure
subplot(4,1,1)
hold on
plot_stretchy_compressy_rule2(median_beta_og)
hold off
subplot(4,1,2)
hold on
histogram(beta_og,beta_bins,'Normalization','pdf','DisplayName','Symmetric beta')
pause(hist_buffer)
xline(median_beta_og,'k--','DisplayName','og median')
xlabel('freq (Hz)')
legend()
title('Parent distribution')
set(gca,'XLim',xlims)
hold off
subplot(4,1,3)
hold on
histogram(beta_compressy,beta_bins,'Normalization','pdf','DisplayName','compressy beta')
pause(hist_buffer)
xline(median_beta_og,'k--','DisplayName',sprintf('og median - %0.2f',median_beta_og))
xline(median_beta_compressy,'r--','DisplayName',sprintf('new median - %0.f',median_beta_compressy))
xlabel('freq (Hz)')
legend()
title('compressy-reg distribution')
set(gca,'XLim',xlims)
hold off
subplot(4,1,4)
hold on
histogram(beta_stretchy,beta_bins,'Normalization','pdf','DisplayName','stretchy beta')
pause(hist_buffer)
xline(median_beta_og,'k--','DisplayName',sprintf('og median - %0.2f',median_beta_og))
xline(median_beta_stretchy,'r--','DisplayName',sprintf('new median - %0.2f',median_beta_stretchy))
xlabel('freq (Hz)')
legend()
title('stretchy-irreg distribution')
set(gca,'XLim',xlims)
hold off
sgtitle('Rule 2 - Symmetric Beta parent distribution')
%% plot uniform distribution before/after
unif_og=unifrnd(2,16,n_samps,1);
%rescale to 2-16 hz range
% unif_og=rescale_nums(unif_og,)
% beta_og=(sim_max-sim_min).*beta_og+sim_min;
median_unif_og=median(unif_og);
mean_unif_og=mean(unif_og);
%RULE 1
[unif_stretchy1,unif_compressy1]=stretchy_compressy_rule1(unif_og);
% get new medians
median_unif_stretchy=median(unif_stretchy1);
median_unif_compressy=median(unif_compressy1);

figure
subplot(4,1,1)
hold on
plot_stretchy_compressy_rule1(median_unif_og)
hold off
subplot(4,1,2)
hold on
histogram(unif_og,'Normalization','pdf','DisplayName','Symmetric unif')
pause(hist_buffer)
xline(median_unif_og,'k--','DisplayName','og median')
xlabel('freq (Hz)')
legend()
title('Parent distribution')
set(gca,'XLim',xlims)
hold off
subplot(4,1,3)
hold on
histogram(unif_compressy1,'Normalization','pdf','DisplayName','compressy unif')
pause(hist_buffer)
xline(median_unif_og,'k--','DisplayName',sprintf('og median - %0.2f',median_unif_og))
xline(median_unif_compressy,'r--','DisplayName',sprintf('new median - %0.2f',median_unif_compressy))
xlabel('freq (Hz)')
legend()
title('compressy-reg distribution')
set(gca,'XLim',xlims)
hold off
subplot(4,1,4)
hold on
histogram(unif_stretchy1,'Normalization','pdf','DisplayName','stretchy unif')
pause(hist_buffer)
xline(median_unif_og,'k--','DisplayName',sprintf('og median - %0.2f',median_unif_og))
xline(median_unif_stretchy,'r--','DisplayName',sprintf('new median - %0.2f',median_unif_stretchy))
xlabel('freq (Hz)')
legend()
title('stretchy-irreg distribution')
set(gca,'XLim',xlims)
hold off
sgtitle('Rule 1 - Symmetric unif parent distribution')

%RULE 2
[unif_stretchy2,unif_compressy2]=stretchy_compressy_rule2(unif_og);
% get new medians
median_unif_stretchy=median(unif_stretchy2);
median_unif_compressy=median(unif_compressy2);

figure
subplot(4,1,1)
hold on
plot_stretchy_compressy_rule2(median_unif_og,{'lin','lin'})
hold off
subplot(4,1,2)
hold on
histogram(unif_og,'Normalization','pdf','DisplayName','Symmetric unif')
pause(hist_buffer)
xline(median_unif_og,'k--','DisplayName','og median')
xlabel('freq (Hz)')
legend()
title('Parent distribution')
set(gca,'XLim',xlims)
hold off
subplot(4,1,3)
hold on
histogram(unif_compressy2,'Normalization','pdf','DisplayName','compressy unif')
pause(hist_buffer)
xline(median_unif_og,'k--','DisplayName',sprintf('og median - %0.2f',median_unif_og))
xline(median_unif_compressy,'r--','DisplayName',sprintf('new median - %0.2f',median_unif_compressy))
xlabel('freq (Hz)')
legend()
title('compressy-reg distribution')
set(gca,'XLim',xlims)
hold off
subplot(4,1,4)
hold on
histogram(unif_stretchy2,'Normalization','pdf','DisplayName','irreg unif')
pause(hist_buffer)
xline(median_unif_og,'k--','DisplayName',sprintf('og median - %0.2f',median_unif_og))
xline(median_unif_stretchy,'r--','DisplayName',sprintf('new median - %0.2f',median_unif_stretchy))
xlabel('freq (Hz)')
legend()
title('stretchy-irreg distribution')
set(gca,'XLim',xlims)
hold off
sgtitle('Rule 2 - Symmetric unif parent distribution')
%% RULE 6
%RULE 2
[unif_stretchy6,unif_compressy6]=stretchy_compressy_rule6(unif_og);
% get new medians
median_unif_stretchy=median(unif_stretchy6);
median_unif_compressy=median(unif_compressy6);

figure
subplot(4,1,1)
hold on
plot_stretchy_compressy_rule6(median_unif_og,{'lin','lin'})
hold off
subplot(4,1,2)
hold on
histogram(unif_og,'Normalization','pdf','DisplayName','Symmetric unif')
pause(hist_buffer)
xline(median_unif_og,'k--','DisplayName','og median')
xlabel('freq (Hz)')
legend()
title('Parent distribution')
set(gca,'XLim',xlims)
hold off
subplot(4,1,3)
hold on
histogram(unif_compressy6,'Normalization','pdf','DisplayName','compressy unif')
pause(hist_buffer)
xline(median_unif_og,'k--','DisplayName',sprintf('og median - %0.2f',median_unif_og))
xline(median_unif_compressy,'r--','DisplayName',sprintf('new median - %0.2f',median_unif_compressy))
xlabel('freq (Hz)')
legend()
title('compressy-reg distribution')
set(gca,'XLim',xlims)
hold off
subplot(4,1,4)
hold on
histogram(unif_stretchy6,'Normalization','pdf','DisplayName','irreg unif')
pause(hist_buffer)
xline(median_unif_og,'k--','DisplayName',sprintf('og median - %0.2f',median_unif_og))
xline(median_unif_stretchy,'r--','DisplayName',sprintf('new median - %0.2f',median_unif_stretchy))
xlabel('freq (Hz)')
legend()
title('stretchy-irreg distribution')
set(gca,'XLim',xlims)
hold off
sgtitle('Rule 6 - Symmetric unif parent distribution')

%% plot log-normal
%% plot irreg cdf cuz I don't believe the median and mean both getting faster in irreg case since mod spec slower
% update: i plotted it and believe it now...
% figure
% cdfplot(warpedPeakFreqs(:,2))
% set(gca,'Xtick',[2 4 6 8 10],'Xscale','log','XLim',xlims)
%% helpers
% function y=extreme_reflect(x,xr,tail_jitter)
%TODO: code this function - thinking of it as a compression towards the
%tails followed by a reflection about the median
% 1. determine how to choose amount of jitter at both extrema (probably
% need more jitter for low freqs than high)
% 2. map to extrema +/- some jitter so not all syllables have the exact
% same rate
% end
function D=get_distribution(distribution)
    switch distribution
        case 'beta'
        case 'uniform'
        case 'data'
            for cc=1:length(conditions)
                cond=conditions{cc};
                condDir=sprintf('%s/%s',warpedStimuliDir,cond);
                Ds=dir([condDir '/*.mat']);
                npeaks_previous=0;
                for dd=1:length(Ds)
                    stimFileName=sprintf('%s/%s',condDir,Ds(dd).name);
                    % read in fs so i dont feel bad about hard coding it even though it
                    % legit does not matter since we've never once changed it 
                    if dd==length(Ds) && cc==length(conditions)
                        [~,fs]=audioread([stimFileName(1:end-4) '.wav']);
                    end
                    load(stimFileName,'s_temp');
                    npeaks_current=size(s_temp,1)-2; %ignore start/end anchorpoints
                    % og intervals only need to be computed once since should be the
                    % same between the two conditions
                    pkIdx=(1:npeaks_current-1)+npeaks_previous;
                    if cc==1
                        ogPeakIntervals(pkIdx,1)=diff(s_temp(2:end-1,1));
                    end
                    warpedPeakIntervals(pkIdx,cc)=diff(s_temp(2:end-1,2));
                    npeaks_previous=pkIdx(end);
                end
            end
            % convert back to time and get frequencies too
            ogPeakIntervals=ogPeakIntervals./fs;
            warpedPeakIntervals=warpedPeakIntervals./fs;
            ogPeakFreqs=1./ogPeakIntervals;
            warpedPeakFreqs=1./warpedPeakIntervals;
            D(1)=ogPeakFreqs;
            D(2)=warpedPeakFreqs;
    end
end

function y=reflect_about(x,xr)
%TODO: it seems like reflection about median giving max values slightly 
% outside og_dist range... why is that? range should not change
    y=x-2.*(x-xr);
end
%TODO:
% function sig=sigmoid(x,k,x0)
%    %TODO: finish sigmoid and play around wik k,x0 
%     arguments
%        x double
%        k =1; % steepness
%        x0= ;
%    end
% end
function rescaled_nums=rescale_nums(nums0,new_lims)
    arguments
        nums0 double
        % note this probably won't work for normalizing to [0,1] ....
        new_lims (1,2) double = [2, 16] 
    end
    rescaled_nums=abs(diff(new_lims)).*nums0+min(new_lims);
end

%RULE 1 with distance-from-median modulated shift
function plot_stretchy_compressy_rule1(f_center,plot_scales,shift_frac,freqs_in,xtix,xlims)
%TODO: make transition at mean for this THEN add the smoothed transiiton
    arguments
        f_center (1,1) double=0
        plot_scales (1,2) = {'log','log'}
        shift_frac=0.4 %not
        freqs_in=1:.5:32
        xtix=2.^(-1:16)
        xlims=[1 34]
        
    end
    [stretchy,compressy]=stretchy_compressy_rule1(freqs_in,f_center,shift_frac);
    plot(freqs_in,stretchy,'b','DisplayName',sprintf('stretchy - %.02f',shift_frac))
    plot(freqs_in,compressy,'r','DisplayName',sprintf('compressy - %.02f',shift_frac))
    % plot(rule_freqs, rule_freqs./(1+shift_frac),'b','DisplayName',sprintf('f/(1+%0.2f)',shift_frac));
    % plot(rule_freqs, rule_freqs.*(1+shift_frac),'b','DisplayName',sprintf('f*(1+%0.2f)',shift_frac));
    plot(freqs_in,freqs_in,'--','DisplayName','Identity Line')
    % xline(f_center,'--','DisplayName','center_f')
    plot(xlims,[f_center,f_center],'--','DisplayName','Og median')
    % plot([f_center, f_center],[1 median_line_max],'--','DisplayName','Median')
    
    xlabel('input freq')
    ylabel('output freq')
    title(sprintf('input-output syllable rate warp rule 1, %s-%s',plot_scales{1},plot_scales{2}))
    set(gca,'Xtick',xtix,'Ytick',xtix, ...
        'Xscale',plot_scales{1},'Yscale',plot_scales{2},'XLim',xlims)
    legend()
end
function [stretchy,compressy]=stretchy_compressy_rule1(og_dist,which_center,max_shift)
% [stretchy,compressy]=stretchy_compressy_rule1(og_dist,which_center,max_shift)

    arguments
        og_dist double
        which_center (1,1) = 0; %0-> median
        max_shift (1,1) double =0.4;
    end
    
    switch which_center
        case -1
            % mean - just use sim dist since should be close to idea anyway
            center_f=mean(og_dist);
        case 0 
            %median
            center_f=median(og_dist);
        case 1
            % mode
            center_f=mode(og_dist);
        otherwise
            center_f=which_center;
    end
    % would a ratio be more consistent with logarithmic shit?
    % maybe something like: dist_factor=og_dist/center_f ...

    
    delta_center=abs(og_dist-center_f);
    % this should be normalized between 0 and 1 but not sure if this is the
    % best way to do it... 
    delta_center=rescale_nums(delta_center,[0, 1]);
    % inc_factor=(1+amt_shift);
    % dec_factor=1./inc_factor;
            
    sim_fast=og_dist>center_f;
    sim_slow=og_dist<center_f;
    
    stretchy=nan(size(og_dist));
    compressy=nan(size(og_dist));
    
    % irreg_dist(sim_fast)=og_dist(sim_fast).*inc_factor;
    % reg_dist(sim_fast)=og_dist(sim_fast).*dec_factor;
    stretchy(sim_fast)=og_dist(sim_fast).*(1+delta_center(sim_fast).*max_shift);
    compressy(sim_fast)=og_dist(sim_fast)./(1+delta_center(sim_fast).*max_shift);

    % irreg_dist(sim_slow)=og_dist(sim_slow).*dec_factor;
    % reg_dist(sim_slow)=og_dist(sim_slow).*inc_factor;
    stretchy(sim_slow)=og_dist(sim_slow)./(1+delta_center(sim_slow).*max_shift);
    compressy(sim_slow)=og_dist(sim_slow).*(1+delta_center(sim_slow).*max_shift);
    
    stretchy(~(sim_slow|sim_fast))=og_dist(~(sim_slow|sim_fast));
    compressy(~(sim_slow|sim_fast))=og_dist(~(sim_slow|sim_fast));
    
    % see if any values crossed over - seems a whole bunch crossing over in
    % reg case, explaining weird bumps in output dist... not sure why since
    % that shouldn't happens
    irreg_fast=stretchy>center_f;
    irreg_slow=stretchy<center_f;
    if any(irreg_fast-sim_fast|irreg_slow-sim_fast)
        sprintf('irreg fast crossovers: %0.0g',sum(abs(irreg_fast-sim_fast)))
        sprintf('irreg slow crossovers: %0.0g',sum(abs(irreg_slow-sim_slow)))
    end

    reg_fast=compressy>center_f;
    reg_slow=compressy<center_f;
     if any(reg_fast-sim_fast|reg_slow-sim_fast)
        sprintf('reg fast crossovers: %0.0g',sum(abs(reg_fast-sim_fast)))
        sprintf('reg slow crossovers: %0.0g',sum(abs(reg_slow-sim_slow)))
    end
end
% %RULE 1
% function plot_stretchy_compressy_rule1(f_center,plot_scales,shift_frac,rule_freqs,xtix,xlims)
%     arguments
%         f_center (1,1) double
%         plot_scales (1,2) = {'log','log'}
%         shift_frac=0.4;
%         rule_freqs=1:32;
%         xtix=2.^(-1:16);
%         xlims=[1 34];
% 
%     end
%     plot(rule_freqs, rule_freqs./(1+shift_frac),'b','DisplayName',sprintf('f/(1+%0.2f)',shift_frac));
%     plot(rule_freqs, rule_freqs.*(1+shift_frac),'b','DisplayName',sprintf('f*(1+%0.2f)',shift_frac));
%     plot(rule_freqs,rule_freqs,'--','DisplayName','Identity Line')
%     xline(f_center,'--','DisplayName','center_f')
%     % plot([f_center, f_center],[1 median_line_max],'--','DisplayName','Median')
% 
%     xlabel('input freq')
%     ylabel('output freq')
%     title(sprintf('input-output syllable rate warp rule 1, %s-%s',plot_scales{1},plot_scales{2}))
%     set(gca,'Xtick',xtix,'Ytick',xtix, ...
%         'Xscale',plot_scales{1},'Yscale',plot_scales{2},'XLim',xlims)
%     legend()
% end
% function [reg_dist, irreg_dist]=stretchy_compressy_rule1(og_dist,which_center,amt_shift)
%     arguments
%         og_dist double
%         which_center (1,1) = 0; %0-> median
%         amt_shift (1,1) double =0.4;
%     end
% 
%     inc_factor=(1+amt_shift);
%     dec_factor=1./inc_factor;
%     switch which_center
%         case -1
%             % mean - just use sim dist since should be close to idea anyway
%             center_f=mean(og_dist);
%         case 0 
%             %median
%             center_f=median(og_dist);
%         case 1
%             % mode
%             center_f=mode(og_dist);
%         otherwise
%             center_f=which_center;
%     end
% 
%     sim_fast=og_dist>center_f;
%     sim_slow=og_dist<center_f;
% 
%     irreg_dist=nan(size(og_dist));
%     reg_dist=nan(size(og_dist));
% 
%     irreg_dist(sim_fast)=og_dist(sim_fast).*inc_factor;
%     reg_dist(sim_fast)=og_dist(sim_fast).*dec_factor;
% 
%     irreg_dist(sim_slow)=og_dist(sim_slow).*dec_factor;
%     reg_dist(sim_slow)=og_dist(sim_slow).*inc_factor;
% 
%     irreg_dist(~(sim_slow|sim_fast))=og_dist(~(sim_slow|sim_fast));
%     reg_dist(~(sim_slow|sim_fast))=og_dist(~(sim_slow|sim_fast));
% 
% end
%RULE 2
function plot_stretchy_compressy_rule2(f_center,plot_scales,shift_amt,rule_freqs,xtix,xlims)
    arguments
        f_center (1,1) double
        plot_scales (1,2) = {'log','log'}
        shift_amt=.3;
        rule_freqs=1:32;
        xtix=2.^(-1:16);
        xlims=[1 34];
        
    end
    plot(rule_freqs, rule_freqs-(1+shift_amt),'b','DisplayName',sprintf('f-%0.2f',shift_amt));
    plot(rule_freqs, rule_freqs+(1+shift_amt),'b','DisplayName',sprintf('f+%0.2f',shift_amt));
    plot(rule_freqs,rule_freqs,'--')
    xline(f_center,'--','DisplayName','center_f')
    % plot([f_center, f_center],[1 median_line_max],'--','DisplayName','Median')
    
    xlabel('input freq')
    ylabel('output freq')
    title(sprintf('input-output syllable rate warp rule 2, %s-%s',plot_scales{1},plot_scales{2}))
    set(gca,'Xtick',xtix,'Ytick',xtix, ...
        'Xscale',plot_scales{1},'Yscale',plot_scales{2},'XLim',xlims)
    legend()
end
function [stretchy,compressy]=stretchy_compressy_rule2(og_dist,which_center,amt_shift)
    arguments
        og_dist double
        which_center (1,1) = 0; %0-> median
        amt_shift (1,1) double =.3;
    end
    
    inc_factor=abs(amt_shift);
    dec_factor=-inc_factor;
    switch which_center
        case -1
            % mean - just use sim dist since should be close to idea anyway
            center_f=mean(og_dist);
        case 0 
            %median
            center_f=median(og_dist);
        case 1
            % mode
            center_f=mode(og_dist); close all
        otherwise
            center_f=which_center;
    end
            
    sim_fast=og_dist>center_f;
    sim_slow=og_dist<center_f;
    
    stretchy=nan(size(og_dist));
    compressy=nan(size(og_dist));
    
    stretchy(sim_fast)=og_dist(sim_fast)+inc_factor;
    compressy(sim_fast)=og_dist(sim_fast)+dec_factor;
    
    stretchy(sim_slow)=og_dist(sim_slow)+dec_factor;
    compressy(sim_slow)=og_dist(sim_slow)+inc_factor;
    
    stretchy(~(sim_slow|sim_fast))=og_dist(~(sim_slow|sim_fast));
    compressy(~(sim_slow|sim_fast))=og_dist(~(sim_slow|sim_fast));

end

%RULE 3 with distance-from-median modulated shift
function plot_stretchy_compressy_rule3(f_center,plot_scales,freqs_in,xtix,xlims)
%TODO: make transition at mean for this THEN add the smoothed transiiton
    arguments
        f_center (1,1) double=0
        plot_scales (1,2) = {'log','log'}
        % compressy_jitter=0.0 %not
        freqs_in=1:.5:32
        xtix=2.^(-1:16)
        xlims=[1 34]
        
    end
    [stretchy,compressy]=stretchy_compressy_rule3(freqs_in,f_center);
    plot(freqs_in,stretchy,'b','DisplayName','stretchy - reflect')
    plot(freqs_in,compressy,'r','DisplayName','compressy - collapse')
    % plot(rule_freqs, rule_freqs./(1+shift_frac),'b','DisplayName',sprintf('f/(1+%0.2f)',shift_frac));
    % plot(rule_freqs, rule_freqs.*(1+shift_frac),'b','DisplayName',sprintf('f*(1+%0.2f)',shift_frac));
    plot(freqs_in,freqs_in,'--','DisplayName','Identity Line')
    % xline(f_center,'--','DisplayName','center_f')
    plot(xlims,[f_center,f_center],'--','DisplayName','Og median')
    % plot([f_center, f_center],[1 median_line_max],'--','DisplayName','Median')
    
    xlabel('input freq')
    ylabel('output freq')
    title(sprintf('input-output syllable rate warp rule 3, %s-%s',plot_scales{1},plot_scales{2}))
    set(gca,'Xtick',xtix,'Ytick',xtix, ...
        'Xscale',plot_scales{1},'Yscale',plot_scales{2},'XLim',xlims)
    legend()
end
function [stretchy,compressy]=stretchy_compressy_rule3(og_dist,which_center,compressy_jitter,clip_lims)
% [stretchy,compressy]=stretchy_compressy_rule1(og_dist,which_center,max_shift)
%TODO: reverse order of reg and irreg and change names to stretchy and
%compressy
    arguments
        og_dist double
        which_center (1,1) = 0; %0-> median
        compressy_jitter (1,1) double =0.0;
        clip_lims=[2 16]

    end
    
    switch which_center
        case -1
            % mean - just use sim dist since should be close to idea anyway
            center_f=mean(og_dist);
        case 0 
            %median
            center_f=median(og_dist);
            % value actually used to produce stimuli - note this doesn't
            % account for discrepancies in peakRate finding algo using the
            % bark envelope yet
            % center_f=5.165503916193135;
        case 1
            % mode
            center_f=mode(og_dist);
        otherwise
            center_f=which_center;
    end
   
    
    stretchy=reflect_about(og_dist,center_f);
    % clip to 2-8 hz
    if any(stretchy>max(clip_lims)|stretchy<min(clip_lims))
        stretchy(stretchy>max(clip_lims))=max(clip_lims);
        stretchy(stretchy<min(clip_lims))=min(clip_lims);
    end
    %TODO: figure out how to add jitter
    compressy=(center_f+compressy_jitter).*ones(size(og_dist));

end
% NOTE SKIPPING RULEXZ 4 AND 5 CUZ SIMPLE AND NOT CONFUSING
%RULE 6 REFLECT WITH COMPRESSION TOWARDS ENDPOINTS
function plot_stretchy_compressy_rule6(f_center,plot_scales,freqs_in,xtix,xlims)
%TODO: make transition at mean for this THEN add the smoothed transiiton
    arguments
        f_center (1,1) double=0
        plot_scales (1,2) = {'log','log'}
        % compressy_jitter=0.0 %not
        freqs_in=1:.5:32
        xtix=2.^(-1:16)
        xlims=[1 34]
        
    end
    [stretchy,compressy]=stretchy_compressy_rule6(freqs_in,f_center);
    plot(freqs_in,stretchy,'b','DisplayName','stretchy - reflect+spread')
    plot(freqs_in,compressy,'r','DisplayName','compressy - collapse')
    % plot(rule_freqs, rule_freqs./(1+shift_frac),'b','DisplayName',sprintf('f/(1+%0.2f)',shift_frac));
    % plot(rule_freqs, rule_freqs.*(1+shift_frac),'b','DisplayName',sprintf('f*(1+%0.2f)',shift_frac));
    plot(freqs_in,freqs_in,'--','DisplayName','Identity Line')
    % xline(f_center,'--','DisplayName','center_f')
    plot(xlims,[f_center,f_center],'--','DisplayName','Og median')
    % plot([f_center, f_center],[1 median_line_max],'--','DisplayName','Median')
    
    xlabel('input freq')
    ylabel('output freq')
    title(sprintf('input-output syllable rate warp rule 6, %s-%s',plot_scales{1},plot_scales{2}))
    set(gca,'Xtick',xtix,'Ytick',xtix, ...
        'Xscale',plot_scales{1},'Yscale',plot_scales{2},'XLim',xlims)
    legend()
end
function [stretchy,compressy]=stretchy_compressy_rule6(og_dist,which_center,compressy_jitter,clip_lims)
% [stretchy,compressy]=stretchy_compressy_rule1(og_dist,which_center,max_shift)
%TODO: reverse order of reg and irreg and change names to stretchy and
%compressy
    arguments
        og_dist double
        which_center (1,1) = 0; %0-> median
        compressy_jitter (1,1) double =0.0;
        clip_lims=[2 16] % Hz

    end
    
    switch which_center
        case -1
            % mean - just use sim dist since should be close to idea anyway
            center_f=mean(og_dist);
        case 0 
            %median
            center_f=median(og_dist);
            % value actually used to produce stimuli - note this doesn't
            % account for discrepancies in peakRate finding algo using the
            % bark envelope yet
            % center_f=5.165503916193135;
        case 1
            % mode
            center_f=mode(og_dist);
        otherwise
            center_f=which_center;
    end
   
    % REFLECT
    stretchy=reflect_about(og_dist,center_f);
    % COMPRESS TOWARDS ENDPOINTS
    d=stretchy-center_f;
    fast=sign(d)==1;
    slow=sign(d)==-1;
    % normalize d to 0,1 range
    d_max_fast=max(clip_lims)-center_f;
    d(fast)=d(fast)/d_max_fast;
    % note: this will be negative so when we divide all d's will be
    % positive
    d_max_slow=min(clip_lims)-center_f;
    d(slow)=d(slow)/d_max_slow;
    stretchy(fast)=(d(fast).*stretchy(fast)+(1-d(fast))*max(clip_lims));
    stretchy(slow)=(d(slow).*stretchy(slow)+(1-d(slow))*min(clip_lims));
    % clip to 2-8 hz
    if any(stretchy>max(clip_lims)|stretchy<min(clip_lims))
        stretchy(stretchy>max(clip_lims))=max(clip_lims);
        stretchy(stretchy<min(clip_lims))=min(clip_lims);
    end
    %TODO: figure out how to add jitter
    compressy=(center_f+compressy_jitter).*ones(size(og_dist));

end