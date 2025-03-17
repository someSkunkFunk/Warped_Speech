
clear, clc
% TODO: add flexibility to mix/match rules across stretchy/compressy
% config.warpedStimuliDir="./median_stretchy_rule1_seg/wrinkle";
config.warpedStimuliDir="./wrinkle/stretchy_compressy";
%NOTE: all should havae same og S matrices but maybe consider switching to
%ones on this dir: .\wrinkle\stretchy_compressy\stretchy_irreg\rule1_seg_bark_median
% NOTE: maintain stretchy THEN compressy order when plotting data
% config.conditions={'reg_stretchy','irreg_stretchy'};
% config.conditions={'stretchy_irreg','compressy_reg'};
% NOTE PLOT RESULTS EXPECTS STRETCHY_IRREG TO BE FIRST FOR LABELS TO BE
% CORRECT... PROBABLY AN EASY FIX SOMEWHERE
config.conditions={'stretchy_irreg','compressy_reg'};
% note: I THINK DIST LIMS USED TO GENERATE TOY DISTRIBUTION SHOULD BE
% DISTINCT FROM CUTOFF APPLIED IN WARP RULES WHICH IS CURRENTLY NOT THE
% CASE
% config.warp_dist_lims=[2 8];
config.sil_tol= 0.75; % in seconds
% config.sil_tol=inf;
config.filter_og=true; % if true, only use intervals within warp_dist_lims range
% config.warp_dist_lims=[1/config.sil_tol 36]; % "empirical" lims
config.warp_dist_lims=[2 8]; % theoretical bounds on syllable rates
config.mod_factor_lims=[3.93328576525152 6.371911573472042]; % where the "stretch" reaches max warp (max_shift) using quartiles
config.sim_dist_lims=[2 16];
config.n_samps=50000; % number of samples in simulated distribution
config.max_shift=0.75;%rule 1(?) and 2 param
config.use_warped_result=false; %toggle between using real results or
% predicted warp distributions - can do both also 
distribution='data'; % 'beta' or 'uniform' or 'data'

config.warp_rule=2; %: 1 - rate shift constant 2 - rate w shift distance modulation
config.stretch_floor=1/0.75; % in hz, slowest allowed output rate
config.rule6pow=2;
% config.jitter=3.3225; %1 std of ogPeakFreqs
config.jitter=[1.75, .25]; % [slow jitter, fast jitter] - sum to 2 Hz
% (assuming this is still sufficient to keep modspec from having weird harmonic peaks)
% config.jitter=0.5;
config.hist_buffer=0.5; % seconds to wait after plotting histogram so lines after it show up on top
config.bin_scale='lin';
config.bin_lims=[.5, 36];
config.n_bins=100;


config.which_center=0;
% NOTE THAT RULE 1/2 PRINT STATEMENTS REGARDING CROSS-OVERS WILL BE PRINTED
% TWICE - ONCE WHEN PLOT WRAPPED IS CALLED AND THEN AGAIN WHEN WARP IS
% CALLED ON STIMULI - WE ONLY CARE ABOUT SECOND ONE BUT DON'T KNOW HOW TO
% MUTE OUTPUT OF FIRST BIT



%% plot warp rule + distribution
D=get_distribution(distribution,config);
medians=median(D,1);
means=mean(D,1);
ogPeakFreqs=D(:,1);
og_quartiles = quantile(ogPeakFreqs, [0.25, 0.5, 0.75]);

%NOTE warpedPeakFreqs number of cols is 2 or 4 depending on
%use_warped_result
warpedPeakFreqs=D(:,2:end);
% 
figure
subplot(size(D,2)+1,1,1)
hold on
plot_warp_rule(medians(1),config)
% NOTE WE DONT ACTUALLY HAVE STIMULI FROM USING RULE 1 YET SO HISTOGRAMS
% SHOW WARPING WITHOUT THE DISTANCE FROM CENTER DISCOUNT
hold off
hist_wrapper(D,config)
sgtitle(sprintf('Warp Rule %d - %s',config.warp_rule,distribution))

warpedDistNms={'og','irreg sim','reg sim','irreg result','reg result'};
for dd=1:size(D,2)
    fprintf('%s median: %0.3g\n',warpedDistNms{dd},medians(dd))
end
%% helpers

function params=parse_params(defaults,varargin)
    params=defaults;
    for pp=1:2:length(varargin)
        params.(varargin{pp})=varargin{pp+1};
    end
end

function hist_wrapper(D,config)
bin_scale=config.bin_scale;
n_bins=config.n_bins;
bin_lims=config.bin_lims;
hist_buffer=config.hist_buffer;
% warp_rule=config.warp_rule;
switch bin_scale
    case 'log'
        freq_bins=logspace(log10(min(bin_lims)),log10(max(bin_lims)),n_bins);% .5-32 Hz ish
    case 'lin'
        freq_bins=linspace(min(bin_lims),max(bin_lims),n_bins);
end
%TODO: detrmine if these vars should be in config
xlims=[1 34];
logTicks=2.^(-1:16);
labels={'og','stretchy - irreg','compressy - reg','stretchy - irreg','compressy - reg'};
titles={'Parent distribution','Irreg Warp Expected output',...
    'Reg Warp Expected output','Irreg Warp result', 'Reg Warp Result'};
for hh=1:size(D,2)
    subplot(size(D,2)+1,1,1+hh)
    hold on
    histogram(D(:,hh),freq_bins,'Normalization','pdf','DisplayName',labels{hh})
    pause(hist_buffer)
    if hh>1
        xline(median(D(:,1)),'k--','DisplayName','og median')
    end
    xline(median(D(:,hh)),'r--','DisplayName','median')
    xlabel('freq (Hz)')
    legend()
    title(titles{hh})
    set(gca,'Xtick',logTicks,'Xscale','log','XLim',xlims)
    hold off
end
end

function D=get_distribution(distribution,config)
% function D=get_distribution(distribution,config)
    sim_dist_lims=config.sim_dist_lims;
    n_samps=config.n_samps;
    conditions=config.conditions;
    % warp_rule=config.warp_rule;
    switch distribution
        case 'beta'
            % params for symmetric beta on[0,1]
            beta_a=4;
            beta_b=4;
            beta_og=betarnd(beta_a,beta_b,n_samps,1);
            % rescale to desired range
            beta_og=rescale_nums(beta_og,sim_dist_lims);
            og_dist_rates=beta_og;
            % [beta_stretchy,beta_compressy]=warp(warp_rule,beta_og);
            % D=[beta_og,beta_stretchy,beta_compressy];

        case 'uniform'
            unif_og=unifrnd(min(sim_dist_lims),max(sim_dist_lims),n_samps,1);
            og_dist_rates=unif_og;
            %rescale to desired range NVM
            % unif_og=rescale_nums(unif_og,dist_lims);
            % [unif_stretchy,unif_compressy]=warp(unif_og,config);
            % D=[unif_og,unif_stretchy,unif_compressy];
        case 'data'
            %NOTE: 
            
            for cc=1:length(conditions)
                cond=conditions{cc};
                if config.which_center~=0
                    error('file directory not specified properly, see note above')
                else
                    % TODO: use a new variable or change how which_center
                    % is utilized in order to toggle between
                    % median,lquartile,uquartile insead of mean/median/mode
                    center_str='median';
                end
                condDir=sprintf('%s/%s/rule%d_seg_bark_%s',config.warpedStimuliDir, ...
                    cond,config.warp_rule,center_str);
                if exist(condDir,"dir")
                    fprintf('loading stimuli from %s\n', condDir)
                    Ds=dir([condDir '/*.mat']);
                    n_ints_previous=0;
                    for dd=1:length(Ds)
                        stimFileName=sprintf('%s/%s',condDir,Ds(dd).name);
                        % get fs from first file in loop, assuming all 
                        % stim across conditions at the same fs
                        if dd==1 && cc==1
                            [~,fs]=audioread([stimFileName(1:end-4) '.wav']);
                        end                    
                        load(stimFileName,'s_temp');
                        % npeaks_current=size(s_temp,1)-2; %ignore start/end anchorpoints
                        % % og intervals only need to be computed once since should be the
                        % % same between the two conditions
                        % interval_indx=(1:npeaks_current-1)+n_ints_previous;
                    
                        %NOTE: Ifrom_temp will be recomputed redundantly for
                        %each condition (and for each stim) but not sure 
                        % how to circumvent this
                        % and not overly time consuming so leaving
                        % inefficiency there
                        Ifrom_temp=diff(s_temp(2:end-1,1));
                        % remove intervals longer than siltol
                        %TODO: check if condition is greater than or equal
                        %to in actual warp script
                        sil_tol_filter=Ifrom_temp<=config.sil_tol*fs;
                        % filter out peakRate vals not within warp_dist_lims
                        if config.filter_og
                            dist_filter=(Ifrom_temp>=min(fs./config.warp_dist_lims)&Ifrom_temp<=max(fs./config.warp_dist_lims));
                        else
                            dist_filter=ones(size(Ifrom_temp));
                        end
                        interval_filter=sil_tol_filter&dist_filter;
                        Ifrom_temp=Ifrom_temp(interval_filter);
                        interval_indx=(1:sum(interval_filter))+n_ints_previous;
                        if cc==1
                            % don't need to assign for each condition,
                            % assuming initial intervals the same in
                            % both...
                            og_dist_intervals(interval_indx,1)=Ifrom_temp;
                        end
                        if config.use_warped_result
                            Ito_temp=diff(s_temp(2:end-1,2));
                            Ito_temp=Ito_temp(interval_filter);
                            warpedPeakIntervals.(cond)(interval_indx,1)=Ito_temp;
                            
                        end
                        n_ints_previous=interval_indx(end);
                    end
                else 
                    fprintf('%s not found - unable to get this data but simulated warping should still work.. \n',condDir)
                    warpedPeakIntervals.(cond)=nan(size(og_dist_intervals));
                end
            end
    
            % convert back to time and get frequencies too
            og_dist_intervals=og_dist_intervals./fs;
            og_dist_rates=1./og_dist_intervals;

            
            % D(:,1)=ogPeakFreqs;
            % D(:,2:1+numel(conditions))=warpedPeakFreqs;
            if config.use_warped_result
                for cc=1:numel(conditions)
                    cond=conditions{cc};
                    warpedPeakIntervals.(cond)=warpedPeakIntervals.(cond)./fs;
                    warpedPeakFreqs.(cond)=1./warpedPeakIntervals.(cond);
                    D(:,3+cc)=warpedPeakFreqs.(cond);
                    % D(:,5)=warpedPeakFreqs.compressy_reg;
                end
            end
    end
    [stretchy,compressy]=warp(og_dist_rates,config);
    D(:,1:3)=[og_dist_rates,stretchy,compressy];
    
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
% function [stretchy,compressy]=warp(warp_rule,og_dist,varargin)
function [stretchy,compressy]=warp(og_dist,config)
% NOTE: varagin and parse params ended up being more cumbersome than config
% so changing how warp and subsequent rule functions handle it to just look
% at config
    switch config.warp_rule
        case 1
            %NOTE:: we might want to edit rule 1 ultimately but for now
            %this should be enough to check if we can at least replicate
            %old functionality with more modularized code
            %RULE 1 with distance-from-median modulated shift
            %NOTE: maybe compressy here should use the inverse of
            %distance-from-median to avoid cross-overs rather than clipping
            %shift to median
            % params=struct('which_center',0,'max_shift',0.4);
            % params=parse_params(params,varargin{:});
            % [stretchy,compressy]=rule1(og_dist,params.which_center,params.max_shift);
            [stretchy,compressy]=rule1(og_dist,config);
        case 2
            %RULE 2 was originally the one with the constant shift to all
            %freqs but I think we ruled that out so just replace with  RULE
            %1 variation where we avoid cross overs in compressy case by
            %using distance-from-median to modulate the shift to each
            %frequency as well as avoiding stretchy shift going outside the
            % legal range using distance from tail rather than
            % hard-clipping
            % params=struct('which_center',0,'max_shift',0.4,'dist_lims',[2 16]);
            % params=parse_params(params,varargin{:});
            % [stretchy,compressy]=rule2(og_dist,params.which_center,params.max_shift);
            [stretchy,compressy]=rule2(og_dist,config);
        case 3
            % params=struct('which_center',0,'dist_lims',[2 16]);
            % params=parse_params(params,varargin{:});
            % [stretchy,compressy]=rule3(og_dist,params.which_center,params.dist_lims);
            [stretchy,compressy]=rule3(og_dist,config);
        case 4 
            % params=struct('which_center',0,'dist_lims',[2 16]);
            % params=parse_params(params,varargin{:});
            % [stretchy,compressy]=rule4(og_dist,params.which_center,params.dist_lims);
            [stretchy,compressy]=rule4(og_dist,config);
        case 5
            % params=struct('which_center',0,'dist_lims',[2 16]);
            % params=parse_params(params,varargin{:});
            % [stretchy,compressy]=rule5(og_dist,params.which_center,params.dist_lims);
            [stretchy,compressy]=rule5(og_dist,config);
        case 6
            [stretchy,compressy]=rule6(og_dist,config);
    end
end
function plot_warp_rule(hline_loc,config,plot_scales,freqs_in,xtix,xlims)
% plot_warp(warp_rule,f_center,plot_scales,freqs_in,xtix,xlims)
%WRAPPER FOR PLOTTING
arguments
    % warp_rule (1,1)
    hline_loc (1,1) double
    config (1,1) struct
    plot_scales (1,2) = {'log','log'}
    freqs_in=1:.5:32
    xtix=2.^(-1:16)
    xlims=[1 36]
    
end
%this seems bad... but didn't change value of which center outside of the
%function so running with it
config.which_center=hline_loc;
%TODO: fix confusing distinction between which_center and f_center
[stretchy,compressy]=warp(freqs_in,config);
% [stretchy,compressy]=warp(warp_rule,freqs_in,"which_center",f_center);
%TODO: make more informative labels for each rule..
plot(freqs_in,stretchy,'b','DisplayName','stretchy - irreg')
plot(freqs_in,compressy,'r','DisplayName','compressy - reg')
plot(freqs_in,freqs_in,'--','DisplayName','Identity Line')
plot(xlims,[hline_loc,hline_loc],'--','DisplayName','Og median')
% plot([f_center, f_center],[1 median_line_max],'--','DisplayName','Median')

xlabel('input freq')
ylabel('output freq')
title(sprintf('input-output syllable rate warp rule %d, %s-%s',config.warp_rule, ...
    plot_scales{1},plot_scales{2}))
set(gca,'Xtick',xtix,'Ytick',xtix, ...
    'Xscale',plot_scales{1},'Yscale',plot_scales{2},'XLim',xlims)
legend()

end
function [stretchy,compressy]=rule1(og_dist,config)
% function [stretchy,compressy]=rule1(og_dist,config)
% function [stretchy,compressy]=rule1(og_dist,which_center,max_shift)
% [stretchy,compressy]=rule1(og_dist,which_center,max_shift)
    % which_center=config.which_center
    
    % arguments
    %     og_dist double
    %     which_center (1,1) %= 0; %0-> median
    %     max_shift (1,1) double %=0.4;
    % end
    which_center=config.which_center;
    max_shift=config.max_shift;
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

    
    % delta_center=abs(og_dist-center_f);
    % this should be normalized between 0 and 1 but not sure if this is the
    % best way to do it... 
    % delta_center=rescale_nums(delta_center,[0, 1]);
    % inc_factor=(1+amt_shift);
    % dec_factor=1./inc_factor;
    shift_rate=1+max_shift;
            
    sim_fast=og_dist>center_f;
    sim_slow=og_dist<center_f;
    
    stretchy=nan(size(og_dist));
    compressy=nan(size(og_dist));
    
    % irreg_dist(sim_fast)=og_dist(sim_fast).*inc_factor;
    % reg_dist(sim_fast)=og_dist(sim_fast).*dec_factor;
    stretchy(sim_fast)=og_dist(sim_fast).*(shift_rate);
    compressy(sim_fast)=og_dist(sim_fast)./(shift_rate);

    % irreg_dist(sim_slow)=og_dist(sim_slow).*dec_factor;
    % reg_dist(sim_slow)=og_dist(sim_slow).*inc_factor;
    stretchy(sim_slow)=og_dist(sim_slow)./(shift_rate);
    compressy(sim_slow)=og_dist(sim_slow).*(shift_rate);
    
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
function [stretchy,compressy]=rule2(og_dist,config)
% [stretchy,compressy]=rule2(og_dist,config)
% function [stretchy,compressy]=rule2(og_dist,which_center,max_shift,dist_lims)
% [stretchy,compressy]=rule2(og_dist,which_center,max_shift)
% similar to rule 1 but with factor accounting for distance from median so
% warp mainly applies to tails of distibution
%TODO: maybe the distance factor should be logarithmic instead of
%linear...?
    % arguments
    %     og_dist double
    %     which_center (1,1) %= 0; %0-> median
    %     max_shift (1,1) double %=0.4;
    %     dist_lims (2,1) double %= [2; 16]
    % 
    % end

    which_center=config.which_center;
    max_shift=config.max_shift;
    % warp_dist_lims
    mod_factor_lims=config.mod_factor_lims;
    
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
    fast=og_dist>center_f;
    slow=og_dist<center_f;
    % pretty sure this wrong cuz point slope formula applies to freq values
    % not their distance from mean vals
    % distance_factor=og_dist-center_f;
    % normalize delta within bounds of distribution such that is scales
    % linearly to 1 at theoretical min/max sylrates
    % distance_factor(fast)=(distance_factor(fast)-max(dist_lims))./(max(dist_lims)-center_f) + 1;
    % distance_factor(slow)=(distance_factor(slow)-min(dist_lims))./(min(dist_lims)-center_f) + 1;
    % distance_factor(~(slow|fast))=0;

    distance_factor=nan(size(og_dist));
    distance_factor(fast)=(1+(og_dist(fast)-max(mod_factor_lims))./(max(mod_factor_lims)-center_f));
    distance_factor(slow)=(1+(og_dist(slow)-min(mod_factor_lims))./(min(mod_factor_lims)-center_f));
    distance_factor(~(slow|fast))=0;
    % don't allow to go past 1
    distance_factor=min(distance_factor,1);
    % modulate beetween 1 (unchanged freq) and 1+max_shift
    shift_rate=1+distance_factor.*max_shift;
    
            
    
    
    stretchy=nan(size(og_dist));
    compressy=nan(size(og_dist));
    
    % irreg_dist(sim_fast)=og_dist(sim_fast).*inc_factor;
    % reg_dist(sim_fast)=og_dist(sim_fast).*dec_factor;
    stretchy(fast)=og_dist(fast).*shift_rate(fast);
    compressy(fast)=og_dist(fast)./shift_rate(fast);

    % irreg_dist(sim_slow)=og_dist(sim_slow).*dec_factor;
    % reg_dist(sim_slow)=og_dist(sim_slow).*inc_factor;
    stretchy(slow)=og_dist(slow)./shift_rate(slow);
    compressy(slow)=og_dist(slow).*shift_rate(slow);
    
    stretchy(~(slow|fast))=og_dist(~(slow|fast));
    compressy(~(slow|fast))=og_dist(~(slow|fast));
    % clip values so they don't go too slow
    stretchy=max(stretchy,config.stretch_floor);
    % clip values within 2-8 hz
    if config.filter_og
        stretchy=max(stretchy,min(config.warp_dist_lims));
        stretchy=min(stretchy,max(config.warp_dist_lims));
    end
    
    % see if any values crossed over - seems a whole bunch crossing over in
    % reg case, explaining weird bumps in output dist... not sure why since
    % that shouldn't happens
    irreg_fast=stretchy>center_f;
    irreg_slow=stretchy<center_f;
    if any(irreg_fast-fast|irreg_slow-fast)
        sprintf('irreg fast crossovers: %0.0g',sum(abs(irreg_fast-fast)))
        sprintf('irreg slow crossovers: %0.0g',sum(abs(irreg_slow-slow)))
    end

    reg_fast=compressy>center_f;
    reg_slow=compressy<center_f;
     if any(reg_fast-fast|reg_slow-fast)
        sprintf('reg fast crossovers: %0.0g',sum(abs(reg_fast-fast)))
        sprintf('reg slow crossovers: %0.0g',sum(abs(reg_slow-slow)))
    end
end
function [stretchy,compressy]=rule3(og_dist,config)
% [stretchy,compressy]=rule3(og_dist,config)

    which_center=config.which_center;
    warp_dist_lims=config.warp_dist_lims;
    
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
    % irreg -> shift "away from" center f by relfecting
    % about
    stretchy=reflect_about(og_dist,center_f);
    % collapse to median (TODO: adds some jitter around center)
    % idea: collapse all to some fast rate to see if that increases high
    % end of mod spec distribution
    compressy=center_f*ones(size(og_dist))+config.jitter.*(2.*rand(size(og_dist))-1);
    %constrain to legal limits
    stretchy(stretchy>max(warp_dist_lims))=max(warp_dist_lims);
    stretchy(stretchy<min(warp_dist_lims))=min(warp_dist_lims);
end

function [stretchy,compressy]=rule4(og_dist,config)
% function [stretchy,compressy]=rule4(og_dist,which_center,dist_lims)
% function [stretchy,compressy]=rule4(og_dist,which_center,dist_lims)
% [stretchy,compressy]=rule4(og_dist,which_center,dist_lims)
% map rates across median to tails of distribution for irreg case, reg case
% just map to center
    % arguments
    %     og_dist double
    %     which_center (1,1) = 0; %0-> median
    %     dist_lims (2,1) double = [2; 16]
    % 
    % end
    which_center=config.which_center;
    warp_dist_lims=config.warp_dist_lims;
    
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
    stretchy=nan(size(og_dist));
    fast=og_dist>center_f;
    slow=og_dist<center_f;
    %NOTE: this should naturally produe output clipped at min/max but
    %isnt...
    stretchy(fast)=min(warp_dist_lims)+config.jitter.*rand(size(stretchy(fast)));
    stretchy(slow)=max(warp_dist_lims)-config.jitter.*rand(size(stretchy(slow)));
    stretchy(~(fast|slow))=center_f;

    compressy=ones(size(og_dist)).*center_f+config.jitter.*(2.*rand(size(og_dist))-1);

end

function [stretchy,compressy]=rule5(og_dist,config)
% [stretchy,compressy]=rule5(og_dist,config)
% function [stretchy,compressy]=rule5(og_dist,which_center,dist_lims)
% map rates to tails of distribution for irreg case, reg case
% just map to center
    % arguments
    %     og_dist double
    %     which_center (1,1) = 0; %0-> median
    %     dist_lims (2,1) double = [2; 16]
    % 
    % end
    
    which_center=config.which_center;
    warp_dist_lims=config.warp_dist_lims;
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
    stretchy=nan(size(og_dist));
    fast=og_dist>center_f;
    slow=og_dist<center_f;
    stretchy(slow)=min(warp_dist_lims)+abs(config.jitter(1)).*rand(size(stretchy(slow)));
    stretchy(fast)=max(warp_dist_lims)-abs(config.jitter(2)).*rand(size(stretchy(fast)));
    stretchy(~(fast|slow))=center_f;

    compressy=ones(size(og_dist));
    % adad random jitter asymetrically
    compressy(slow)=compressy(slow).*center_f-config.jitter(1).*(rand(size(og_dist(slow))));
    compressy(fast)=compressy(fast).*center_f+config.jitter(2).*(rand(size(og_dist(fast))));
    compressy(~(fast|slow))=center_f;
end
function [stretchy,compressy]=rule6(og_dist,config)
% [stretchy,compressy]=rule2(og_dist,config)
% function [stretchy,compressy]=rule2(og_dist,which_center,max_shift,dist_lims)
% [stretchy,compressy]=rule2(og_dist,which_center,max_shift)
% similar to rule 1 but with factor accounting for distance from median so
% warp mainly applies to tails of distibution
%TODO: maybe the distance factor should be logarithmic instead of
%linear...?
    % arguments
    %     og_dist double
    %     which_center (1,1) %= 0; %0-> median
    %     max_shift (1,1) double %=0.4;
    %     dist_lims (2,1) double %= [2; 16]
    % 
    % end

    which_center=config.which_center;
    max_shift=config.max_shift;
    warp_dist_lims=config.warp_dist_lims;
    pow=config.rule6pow;
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
    fast=og_dist>center_f;
    slow=og_dist<center_f;
    % pretty sure this wrong cuz point slope formula applies to freq values
    % not their distance from mean vals
    % distance_factor=og_dist-center_f;
    % normalize delta within bounds of distribution such that is scales
    % linearly to 1 at theoretical min/max sylrates
    % distance_factor(fast)=(distance_factor(fast)-max(dist_lims))./(max(dist_lims)-center_f) + 1;
    % distance_factor(slow)=(distance_factor(slow)-min(dist_lims))./(min(dist_lims)-center_f) + 1;
    % distance_factor(~(slow|fast))=0;

    distance_factor=nan(size(og_dist));
    distance_factor(fast)=(1+(og_dist(fast)-max(warp_dist_lims))./(max(warp_dist_lims)-center_f)).^pow;
    distance_factor(slow)=(1+(og_dist(slow)-min(warp_dist_lims))./(min(warp_dist_lims)-center_f)).^pow;
    distance_factor(~(slow|fast))=0;
    shift_rate=1+distance_factor.*max_shift;
    
            
    
    
    stretchy=nan(size(og_dist));
    compressy=nan(size(og_dist));
    
    % irreg_dist(sim_fast)=og_dist(sim_fast).*inc_factor;
    % reg_dist(sim_fast)=og_dist(sim_fast).*dec_factor;
    stretchy(fast)=og_dist(fast).*shift_rate(fast);
    compressy(fast)=og_dist(fast)./shift_rate(fast);

    % irreg_dist(sim_slow)=og_dist(sim_slow).*dec_factor;
    % reg_dist(sim_slow)=og_dist(sim_slow).*inc_factor;
    stretchy(slow)=og_dist(slow)./shift_rate(slow);
    compressy(slow)=og_dist(slow).*shift_rate(slow);
    
    stretchy(~(slow|fast))=og_dist(~(slow|fast));
    compressy(~(slow|fast))=og_dist(~(slow|fast));
    
    % see if any values crossed over - seems a whole bunch crossing over in
    % reg case, explaining weird bumps in output dist... not sure why since
    % that shouldn't happens
    irreg_fast=stretchy>center_f;
    irreg_slow=stretchy<center_f;
    if any(irreg_fast-fast|irreg_slow-fast)
        sprintf('irreg fast crossovers: %0.0g',sum(abs(irreg_fast-fast)))
        sprintf('irreg slow crossovers: %0.0g',sum(abs(irreg_slow-slow)))
    end

    reg_fast=compressy>center_f;
    reg_slow=compressy<center_f;
     if any(reg_fast-fast|reg_slow-fast)
        sprintf('reg fast crossovers: %0.0g',sum(abs(reg_fast-fast)))
        sprintf('reg slow crossovers: %0.0g',sum(abs(reg_slow-slow)))
    end
end