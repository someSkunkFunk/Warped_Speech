function [wf_warp,s] = stretchyWrinkle(wf,fs,k,center,rule,shift_rate, ...
    peak_tol,sil_tol,min_mod_interval,max_mod_interval,env_method,jitter, ...
    interval_ceil_out)%,input_syll_rate_lims,output_syll_rate_lims)
% [wf_warp,s] = stretchyWrinkle(wf,fs,k,center,rule,shift_rate, ...
    % peak_tol,sil_tol,minInt,maxInt,env_method,jitter,interval_ceil_out,),...
    % syll_rate_lims)
%RULE2 TODOS:
%  make longest intervals/slowest freqs actually correspond with sil tol
%  make fast freqs/ short intervals increase faster (perhaps by decreasing
%       ramp-up maximum point)
%  de-couple ramp-up factor from frequency past the extrema (should be 1
%  for freqs beyond min/max)



%[wf_warp,s] = stretchyWrinkle(wf,fs,k,center,rule,shift_rate,peak_tol,sil_tol,minInt,maxInt,env_method,jitter)
% k: 1->make regular, -1-> more irregular
% center: 1 -> mean, 2 -> median, 3 -> mode
% rule: 
% 1 -> div/multiply freq to stretch/compress      
% 2->add/subtract fixed amt from freq to stretch/compress
% 3-> rerflect about/map to median to "stretch" compress
% question: center can be measured on-line for current stimulus, or
% hard-coded as mean/median/mode of pre-computed distribution - idk which
% one is more valid
% other considerations: can operate on the syllable intervals directly or
% the syllable rate then invert the intervals - maybe worthwhile trying
% both
% first implementation will use central measure considering all the
% intervals - including silent pauses ("raw syllable rate")
% alternatively can calculate using only continuous speech segments
% ("articulation rate")
% TODO: MAKE SURE TO ENFORCE SIL_TOL > MAXiNT TO AVOID ERRORS
% RE MIN/MAX INTS: FILTERING BASED ON INTERVAL LENGTH WON'T WORK THE WAY 
% WE WANT - FOR NOW THINKING THAT I PROBABLY DOESN'T MATTER AS LONG AS THE
% WARPING PARAMTERS ARE BASED ON MANIPULATING JUST THOSE INTERVALS WITHIN
% THE SYLLABIC RANGE... MAYBE THAT'S ALSO FLAWED THINKING BUT IM PRAYING
% IT'S NOT; THIS ALSO PROBABLY AFFECTS HOW WE CALCULATE THE DISTANCE FROM
% MEDIAN DISCOUNT BUT FOR NOW I'M JUST GONNA CLIP IT AT 1 FOR VALUES
% OUTSIDE THE SYLLABIC RANGE CUZ I DONT HAVE TIME TO THINK ABOUT WHAT THE
% RIGHT THING TO DO IS
arguments
    wf double
    fs (1,1) double
    k (1,1) double = 1;
    center (1,1) double = 0; %TODO: use empirical value 5.8644
    rule (1,1) = 1;    
    %TODO: figure out what happens when we make this smaller or larger
    % shift_rate (1,1) double = 0.40; % percentage of original interval - NOTE: should consider spitting this variable out if we start varying it in future - i think once we get rid of long pauses in the estimation that won't really be necessary
    shift_rate (1,1) double = 1.00; 
    peak_tol (1,1) double = 0.1;
    %we checked that sil_tol=inf replicates original unsegmented results
    %but now we also gotta remember that sil_tol needs to be larger than
    %maxInt otherwise seg indexing won't match Ifrom
    sil_tol (1,1) double = 0.75;
    % for test to verify results match what we got without worrying about
    % silent segments
    % sil_tol (1,1) double = inf
    % min_mod_interval (1,1) double = 0.0625; %16 hz
    % min_mod_interval (1,1) double = 0.125; %8 hz
    % min_mod_interval (1,1) double = 1/36;
    % min_mod_interval (1,1) double = 1/7; %(1/ Hz) % where the "stretch" reaches max warp (shift_rate) for fast syllables
    % config.mod_factor_lims=[3.93328576525152 6.371911573472042]; upper
    % and lower quartiles ^
    % min_mod_interval (1,1) double =1/6.371911573472042; %1/5.5;%
    % max_mod_interval (1,1) double = 1/3.93328576525152; %1/2.5;
    % max_mod_interval (1,1) double = 0.5; % 2 hz
    % max_mod_interval (1,1) double = 1/2;
    % max_mod_interval (1,1) double = 0.75; %1.33 Hz-> 
    % max_mod_interval (1,1) double = 1/3;%1/hz where the "stretch" reaches max warp (shift_rate) for slow syllables
    % max_mod_interval (1,1) double = 100000/1; %.000001 hz
    % Quantiles (45%,55% in Hz): 4.320, 4.927 (prominence,width - 0.105, 1.842)
    min_mod_interval (1,1) double = 1/4.927;
    % max_mod_interval (1,1) double = 1/4.32;
    %trying something a bit slower (25% quantile) cuz too many slow syllables:
    % max_mod_interval (1,1) double = 1/3.186;
    % that wasn't low enough, trying some random number:
    max_mod_interval (1,1) double = 1/2.0;
    env_method (1,1) double = 2 % 1-> use broadband 2-> use bark filterbank 3-> use gammatone filterbank (TODO)
    % jitter=3.3225; % 1 std of ogPeakFreqs... must do rules 3 and up reg operation in freq domain then convert to time
    jitter=[1.75, .25]; %in Hz - slow jitter, fast jitter - should add to 2 hz
    interval_ceil_out (1,1) double =0.75; %in s, maximum output interval
    % only treat as syllables if within syll_rate_limits - filters input
    % and output peakRate vals

    %NOTE: can probably use the same set of values for stuff below since
    %theoretically represent intervals at extrema of possible syllable
    %rates
    % input_syll_rate_lims (1,2) double = [2, 8]; % in Hz
    % output_syll_rate_lims (1,2) double = [2, 8]; % in Hz


end
switch center 
        case -1 
            % use lower quartile
            % estimated from empirical distribution (bark-envelope)
            f_center=4.12960014982676;
            error('value not updated for constrained distribution')
            % use mean
            % calculated using product of gammafit a,b params: 3.2751, 2.0123 -> mean freq: 6.5905 hz (0.15 s)
            % f_center=6.5905;% calculated on untruncated distribution
        case 0 % use median
            % estimated from median of 1e5 gamma samples using a,b: 3.2751, 2.0123
            % realizing this one should probably also be estimated with
            % rejection resampling since even if using entire range of
            % intervals to model distribution, there is not an unbounded
            % 0-inf range on intervals....
            % f_center=5.9276;
            % median from using bark-env envelope gamma pdf fit 
            % (a,b : 5.47, 1.11) 2-8Hz clipping
            % f_center=5.1412;
            % empirical median from bark-envelope sylrate (over entire freq range):
            % f_center=5.86436170212766;
            % empirical median from bark-envelope sylrate (filtered between 2-8 Hz):
            % f_center=5.09532062391681;
            % median: 4.626 (prominence,width - 0.105, 1.842)
            f_center=4.626;
        case 1 
            % upper quartile
            % estimated from empirical distribution (bark-env)
            f_center=8.21688093907211;
            error('value not updated for constrained distribution')
            % use mode
            % f_center=4.5782;% calculated on untruncated distribution
        otherwise
            % explicitly provide a specific numeric value
            f_center=center;
end

% For repeatability
rng(1);


% Extract envelope

%TODO: determine if bark_env function i took from "pushing the envelope"
%used that specific gaussian filter implementation instead of what they
%said they did in the paper (10 Hz lowpass) for some performance reason
%that will cause our lowpass filter to give bad results....?
switch env_method
    
    case 1 % use broadband envelope
        env = abs(hilbert(wf));
    case 2 % use bark scale filterbank
        %TODO: saw something in their code about removing spurious peaks
        %and rescaling to some common range (i think they used -1 to 1) -
        %we should investigate if incorporating those steps improves the
        %output (after comparing the syllable rate output distribution
        %using this method vs the broadband and GC envelope
        env=bark_env(wf,fs,fs);
    case 3
        %TODO: test this shit
        env=extractGCEnvelope(wf,fs);

end
%% STUFF MODED TO DEDICATED FUNCTION %
% Hd = getLPFilt(fs,10); %% Maybe don't filter so harshly?
% env = filtfilthd(Hd,env);
% % Find onsets
% env_onset = diff(env);
% env_onset(env_onset<0) = 0;
% 
% [~,Ifrom,~,p] = findpeaks(env_onset,fs);
% 
% % normalize prominence
% p = p./std(p);
% Eliminate small peaks
% Ifrom(p<peak_tol)=[];
% p(p<peak_tol)=[];
[Ifrom,~,p,w,~]=get_peakRate(env,fs,peak_tol);
%% FILTER PEAKS HERE
% set thresholds
p_t=0.105;
w_t=2.026;
Ifrom=Ifrom(p>p_t&w>w_t);
%%


% eliminate stuff outside range where syllables happen 
%TODO: that's not gonna work probably there's some filtering we wanna
%modify in the peakfinding algo instead cuz just filtering based on the
%diff doesn't guarantee not a syllable but i gotta think about this more
%carefully
% Ifrom(Ifrom>maxInt|Ifrom<minInt)=[];
% p(Ifrom>maxInt|Ifrom<minInt)=[];
%TODO: ask aaron why he had preallocated 3 different Ito...'s when one would suffice (only one used to warp ultimately?)
% [ItoReg,ItoIrreg] = deal(zeros(size(Ifrom))); 
% Ito=zeros(size(Ifrom));
% get interpeak intervals
% IPI0=diff(Ifrom);
seg=[[1; find(diff(Ifrom)>sil_tol)+1] [find(diff(Ifrom)>sil_tol); length(Ifrom)]];
%inter-segment interval
ISI=0;

% invert to get articulation rates
% IPF0=1./IPI0;
% IPF1=zeros(size(IPF0));
% slow=IPF0<f_center;
% fast=IPF0>f_center;
%NOTE: none should be exactly equal to the center f but maybe include a
%thing just in case
% IPF1(~(slow|fast))=IPF0(~(slow|fast));
n_segs=size(seg,1);
for ss=1:n_segs 
    % % get rates for current segment
    % IPF0_seg=IPF0(seg(ss,1):seg(ss,2));
    % IPF1_seg=IPF1_seg(seg(ss,1):seg(ss,2));
    % get intervals for current segment
    IPI0_seg=diff(Ifrom(seg(ss,1):seg(ss,2)));
    % constrain to syl_rate_limits %NOTE this won't actually work...
    % IPI0_seg=max(IPI0_seg,min(1./input_syll_rate_lims));
    % IPI0_seg=min(IPI0_seg,max(1./input_syll_rate_lims));
    % IPI1_seg=IPI1(seg(ss,1):seg(ss,2));
    % slow=IPF0_seg<f_center;
    % fast=IPF0_seg>f_center;
    % 
    slow=1./IPI0_seg<f_center;
    fast=1./IPI0_seg>f_center;
    % median values don't change in either rule
    % IPF1_seg(~(slow|fast))=IPF0_seg(~(slow|fast));
    IPI1_seg=nan(size(IPI0_seg));
    IPI1_seg(~(slow|fast))=IPI0_seg(~(slow|fast));
    switch rule
        case 1
            % RULE 1
            % % multiply/divide by fixed ratio of input rate
            % rate_shift=(1+abs(amt_shift));
            % multiply/divide by distance-from-median dependent ratio of input rate
            % TODO: this was a mid attempt at modulating the shift amount
            % based on distance from center_f so that when center_f was
            % median, the median would not change after warping. 
            % IPI0_seg_clipped=IPI0_seg;
            % IPI0_seg_clipped(IPI0_seg_clipped>1/minInt)=1/minInt;
            % IPI0_seg_clipped(IPI0_seg_clipped<1/maxInt)=1/maxInt;
            % dist_discount=abs(IPI0_seg_clipped-f_center);
            % normalize range... TODO: is there a more intelligent way to
            % do this?
            %TODO: update this with range determined by distribution cutoff
            %frequencies (or the peakrate detection cutoffs? not sure why
            %they'd be different...)
            %ERROR PROBABLY COMING FROM HERE            
            % dist_discount=(dist_discount-1/maxInt)./syl_rate_dist_range;
            % rate_shift=(1+dist_discount.*abs(amt_shift));
            rate_shift=(1+abs(shift_rate));

            switch k
                case 1
                    % reg -> shift towards center f
                    % IPF1_seg(slow)=IPF0_seg(slow).*rate_shift;
                    % IPF1_seg(fast)=IPF0_seg(fast)./rate_shift;

                    % this is equivalent to above when working in 
                    % time domain cuz reciprocals:
                    IPI1_seg(slow)=IPI0_seg(slow)./rate_shift;
                    IPI1_seg(fast)=IPI0_seg(fast).*rate_shift;

                    % this keeps percent change constant but probably not what we want
                    % IPF1(fast)=IPF0(fast).*(1-amt_shift);
                case -1
                    % irreg -> shift away from center f
                    % IPF1_seg(slow)=IPF0_seg(slow)./rate_shift;
                    % IPF1_seg(fast)=IPF0_seg(fast).*rate_shift;
                    
                    % this is equivalent to above when working in 
                    % time domain cuz reciprocals:
                    IPI1_seg(slow)=IPI0_seg(slow).*rate_shift;
                    IPI1_seg(fast)=IPI0_seg(fast)./rate_shift;

                    % this keeps percent change constant but probably not what we want
                    % IPF1(slow)=IPF0(slow).*(1-amt_shift);
            end

        case 2
            % RULE 2
            % % multiply/divide by fixed ratio of input rate scaled by
            % distance from median 
            % distance_factor=nan(size(og_dist));
            % distance_factor(fast)=1+(og_dist(fast)-max(dist_lims))./(max(dist_lims)-center_f);
            % distance_factor(slow)=1+(og_dist(slow)-min(dist_lims))./(min(dist_lims)-center_f);
            % distance_factor(~(slow|fast))=0;
            % rate_shift=1+distance_factor.*max_shift;
            % NOTE: converting to freqs even though working in time domain
            % because i figured out the rule in freq domain while drunk at
            % a bar and don't feel like doing that extra step
            mod_lims=[1/min_mod_interval 1/max_mod_interval];
            dist_factor=nan(size(IPI0_seg));
            dist_factor(fast)=1+(1./IPI0_seg(fast)-max(mod_lims))./(max(mod_lims)-f_center);
            dist_factor(slow)=1+(1./IPI0_seg(slow)-min(mod_lims))./(min(mod_lims)-f_center);
            dist_factor(~(slow|fast))=0;
            % constrain modulation factor below 1
            dist_factor=min(dist_factor,1);


            
            if any(dist_factor<0)
                error('this is fucked.')
            end
            % modulate from 1 (no change) to shift_rate+1
            rate_shift=(1+dist_factor.*shift_rate);
            
            switch k
                % NOTE: working in time domain
                case 1
                    % reg -> shift towards center f
                    % slow intervals get faster (shorter)
                    IPI1_seg(slow)=IPI0_seg(slow)./rate_shift(slow);
                    % fast intervals get slower (longer)
                    IPI1_seg(fast)=IPI0_seg(fast).*rate_shift(fast);
                case -1
                    % irreg -> shift away from center f
                    % slow intervals get slower (longer)
                    IPI1_seg(slow)=IPI0_seg(slow).*rate_shift(slow);
                    % fast intervals get faster (shorter)
                    IPI1_seg(fast)=IPI0_seg(fast)./rate_shift(fast);
    
            end
            
            
        case 3
        % RULE 3
        % relfect about/collapse to median
        switch k
            case 1
                % reg -> shift towards center f
                IPI1_seg(slow)=1./(f_center+jitter.*(2.*rand(size(IPI0_seg(slow)))-1));
                IPI1_seg(fast)=1./(f_center+jitter.*(2.*rand(size(IPI0_seg(fast)))-1));
            case -1
                % irreg -> "invert" cadence by making fast stuff slow and
                % vice verse via reflection about f_center
                IPI1_seg=1./reflect_about((1./IPI0_seg),f_center);
                % IPI1_seg(slow)=1./(IPI0_seg(slow).^-1-amt_shift2);
                % IPI1_seg(fast)=1./(IPI0_seg(fast).^-1+amt_shift2);
           %todo: think about what to do with potential negative
           %values...
           if any(IPI1_seg<min_mod_interval)
               % error('what do')
               % map to minInt
               %TODO: is there a reason this might be a bad idea?
               % - a bunch of stuff gets mapped to 2 hz but also below
               % that... gonna clip at 2 hz tho probably just gonna
               % make a hump at 2 hz...
               IPI1_seg(IPI1_seg<min_mod_interval)=min_mod_interval;
           end
           %temporary fix to overly elongated syllables
           if any(IPI1_seg>max_mod_interval)
                IPI1_seg(IPI1_seg>max_mod_interval)=max_mod_interval;
           end
        end
    case 4
    % RULE 4
    % all values cross median to tails of distribution
    
        switch k
            case 1
                % reg -> shift towards center f
                IPI1_seg(slow)=1./(f_center+jitter.*(2.*rand(size(IPI0_seg(slow)))-1));
                IPI1_seg(fast)=1./(f_center+jitter.*(2.*rand(size(IPI0_seg(fast)))-1));
            case -1
                % irreg -> fast go to slow maximally and slow go to fast
                % maximally
                % note: 1 std of jitter here seems too broad.. also applies
                % in rule 5 irreg
                IPI1_seg(slow)=1./(1/min_mod_interval-jitter.*(rand(size(IPI0_seg(slow)))));
                IPI1_seg(fast)=1./(1/max_mod_interval+jitter.*(rand(size(IPI0_seg(fast)))));
        end

    case 5
        % RULE 5 map fast stuff to fastest allowable rate and slow stuff to
        % slowest allowable rate
         switch k
            case 1
                % reg
                IPI1_seg(slow)=1./(f_center-abs(jitter(1)).*rand(size(IPI0_seg(slow))));
                IPI1_seg(fast)=1./(f_center+abs(jitter(2)).*rand(size(IPI0_seg(fast))));
                IPI1_seg(~(fast|slow))=1./f_center;
            case -1
                % irreg -> fast go to fast maximally and slow go to slow
                % maximally
                IPI1_seg(slow)=1./(1/max_mod_interval+jitter.*(rand(size(IPI0_seg(slow)))));
                IPI1_seg(fast)=1./(1/min_mod_interval-jitter.*(rand(size(IPI0_seg(fast)))));
        end
    end

    if ss>1
        ISI=Ifrom(seg(ss,1))-Ifrom(seg(ss,1)-1);
        start_t=Ito(end)+ISI;
    else
        %won't ISI just be zero for first segment....?
        start_t=Ifrom(seg(ss,1))+ISI;
    end
    % enforce maximum interval/minimum freq allowed in
    % output
    IPI1_seg=min(IPI1_seg,interval_ceil_out);
    %NOTE: (below) still work in progress... likely don't need separate
    %sylrate limits for inout and output
    % IPI1_seg=max(IPI1_seg,min(1./output_syll_rate_lims));
    % IPI1_seg=min(IPI1_seg,max(1./output_syll_rate_lims));
    
    %CUMSUM
    Ito(seg(ss,1):seg(ss,2),1)=[start_t; start_t+cumsum(IPI1_seg)];
    if any(isnan(Ito))
        error('wtf duude.')
    end
    % IPF1(seg(ss,1):seg(ss,2))=IPF1_seg;
    % invert segment rates back to time intervals
    % IPI1_seg=1./IPF1_seg;
    
end
% % invert back to time
% IPI1=1./IPF1;
%MAYBE THIS IS THE ERROR?
%Ifrom(1) is time of first peak...
% Ito=[Ifrom(1);Ifrom(1)+cumsum(IPI1)];
% convert to indices
s = round([Ifrom Ito]*fs);
% pad indices with correct start/end times
%TODO: last index is manually being set to length of wf but audio clips on
%output are definitely not the same duration as the original so wtf is
%going on here dude???
% NVM I think it's the following line where the last value of s in output
% column gets re-written to be whatever the penultimate was plus the silent
% bit at the end of original
s = [ones(1,2); s; ones(1,2)*length(wf)];
% fix last value to 
end_sil = diff(s(:,1)); end_sil = end_sil(end);
s(end,2) = s(end-1,2)+end_sil;
% warp
param.tolerance = 256;
param.synHop = 256;
param.win = win(1024,2); % hann window
% try
%     wf_warp = wsolaTSM(wf,s,param);
% catch
%     ;
% end
wf_warp = wsolaTSM(wf,s,param);
end
function y=reflect_about(x,xr)
    y=x-2.*(x-xr);
end


%% Fossils
% 
% % env derivative peak rate (measured - 1/mean(intervals))
% 
% f = 6.3582;
% % f=5.6256; %if using 1/mean(intervals)
% % f=6.9457; %if using mean(1/intervals)
% 
% % For repeatability
% rng(1);
% 
% %% HI ANDRE 250
% % We need TSM toolbox - should already be added in warp script tho
% % addpath(sprintf('%s/TSM Toolbox',boxfolder))
% %TODO: verify this follows same logic we used to generate distribution
% % Extract envelope
% Hd = getLPFilt(fs,10); %% Maybe don't filter so harshly?
% env = abs(hilbert(wf));
% env = filtfilthd(Hd,env);
% 
% % Find onsets
% env_onset = diff(env);
% env_onset(env_onset<0) = 0;
% 
% [~,Ifrom,~,p] = findpeaks(env_onset,fs);
% 
% % normalize prominence
% p = p./std(p);
% 
% % Eliminate small peaks
% Ifrom(p<peak_tol)=[];
% p(p<peak_tol)=[];
% 
% seg = [[1; find(diff(Ifrom)>sil_tol)+1] [find(diff(Ifrom)>sil_tol); length(Ifrom)]];
% 
% 
% % inter-segment interval
% % updated each segment iteration to preserve timing between segments
% ISI = 0;
% [ItoReg,ItoShuff,ItoRand] = deal(zeros(size(Ifrom)));
% for ss = 1:size(seg,1)
%     % Get the intervals for the current segment
%     IPI = diff(Ifrom(seg(ss,1):seg(ss,2)));
% 
%     % decide regularity
%     switch sign(k)
%         case -1 % Make more irregular
%             if numel(IPI)>1 % if at least 3 peaks (2 intervals)
%                 nsamps = diff(seg(ss,:));
%                 secs = Ifrom(seg(ss,:));
%                 switch rand_how
%                     case 1
%                         I = gammaRandInterval(nsamps,secs);
%                     case 2
%                         I = unifRandInterval(nsamps,secs);
%                 end
%                 ItoRand(seg(ss,1):seg(ss,2)) = I;
%             else % else just use the original peaks
%                 ItoRand(seg(ss,1):seg(ss,2)) = Ifrom(seg(ss,1):seg(ss,2));
%             end
% 
%         case 1 % Make more regular
% %             ItoReg(seg(ss,1):seg(ss,2)) = linspace(Ifrom(seg(ss,1)),Ifrom(seg(ss,2)),diff(seg(ss,:))+1);
% 
%             if ss>1
%                 ISI = Ifrom(seg(ss,1))-Ifrom(seg(ss,1)-1);
%                 start_t = ItoReg(find(ItoReg,1,'last'))+ISI;
%             else
%                 start_t = Ifrom(seg(ss,1))+ISI;
%             end
% 
%             ItoReg(seg(ss,1):seg(ss,2)) = start_t+(0:diff(seg(ss,:)))./f;
% 
%         case 0 % Shuffle intervals
%             ItoShuff(seg(ss,1):seg(ss,2)) = Ifrom(seg(ss,1))+[0; cumsum(IPI(randperm(length(IPI))))];
% 
%         otherwise
%             if isnan(k)
%                 ItoReg(seg(ss,1):seg(ss,2)) = linspace(Ifrom(seg(ss,1)),Ifrom(seg(ss,2)),diff(seg(ss,:))+1);
%             else
%                 error('wrong')
%             end
%     end
%     % fprintf('%d\n',ii)
% 
% 
% 
%     % ItoRand = Ifrom+rand_jitter.*randn(size(Ifrom));
%     % while min(diff(ItoRand))<0.02
%     % ItoRand = Ifrom+rand_jitter.*randn(size(Ifrom));
%     % end
% 
%     % Ole's idea: define each word's random interval by finding midpoint between words?
% 
%     % Ed's idea: Add jitter as a percentage of original interval.
% 
% end
% 
% 
% switch sign(k)
%     case -1
%         Ito = Ifrom+(ItoRand-Ifrom).*abs(k);
%     case 1
%         Ito = Ifrom+(ItoReg-Ifrom).*abs(k);
%     case 0
%         Ito = ItoShuff;
%     otherwise
%         Ito = Ifrom+(ItoReg-Ifrom);
% end
% 
% s = round([Ifrom Ito]*fs);
% s = [ones(1,2); s; ones(1,2)*length(wf)];
% end_sil = diff(s(:,1)); end_sil = end_sil(end);
% s(end,2) = s(end-1,2)+end_sil;
% 
% param.tolerance = 256;
% param.synHop = 256;
% param.win = win(1024,2); % hann window
% wf_warp = wsolaTSM(wf,s,param);
% 
% % Re-mixed a re-mix and it was back to normal.
% if isnan(k)
%     s = fliplr(s);
%     wf_warp = wsolaTSM(wf_warp,s,param);
% end
% 
% % Plotting
% if 0
%     env_fs = 2048; 
%     env_warp = abs(hilbert(wf_warp));
%     env_warp = filtfilthd(Hd,env_warp);
% 
%     env = resample(env,env_fs,fs);
%     env_warp = resample(env_warp,env_fs,fs);
% 
%     t_env = linspace(0,length(env)./env_fs,length(env));
% 
%     figure
%     ex(1) = subplot(2,1,1);
%     plot(t_env,env./std(env))
% 
% 
%     ex(2) = subplot(2,1,2);
%     plot(t_env,env_warp./std(env_warp))
%     linkaxes(ex,'x')%
% end
% 
% end
% 
% function I = quantizedRand(seg,Q,Ifrom,wav_fs)
% 
% samps = round(Ifrom(seg(ss,1))*wav_fs):Q*wav_fs:round(Ifrom(seg(ss,2))*wav_fs);
% I = sort(datasample(samps,diff(seg(ss,:))+1,'Replace',false));
% I([1 end]) = round(Ifrom(seg(ss,:))*wav_fs);
% 
% while min(diff(I))>(0.04*wav_fs)&&max(diff(I))<(0.3*wav_fs)
%     I = sort(datasample(samps,diff(seg(ss,:))+1,'Replace',false));
%     I([1 end]) = round(Ifrom(seg(ss,:))*wav_fs);
% end
% I = I./wav_fs;
% end
% 
% 
% function [I,ii] = gammaRandInterval(nsamps,secs,a,b)
% 
% arguments
%     nsamps (1,1) double
%     secs (2,1) double
%     % a = 3.7;
%     % b = 2;
%     % original distribution parameters estimated by aaron
%     % a=6.3;
%     % b=1.05;
%     % original distribution parameters estimated by truncated mle gamma
%     % fit:
%     % a=4.2;
%     % b=1.6;
%     % updated MLE fit params from lowering upper frequency cutoff to 8 hz
%     a=5.2781;
%     b=1.1751;
% end
% 
% buff = 0.04; %should buffer depend on segment duration?
% % per Poeppel & Assaneo 2-8Hz range consistent across languages/speakers
% minInt = 0.125;
% maxInt = 0.5;
% Icrit = 100000;
% 
% % Original distribution
% % a=6.3;
% % b=1.05;
% 
% % broader distribution
% % a = 3.7;
% % b = 2;
% 
% ii = 0;
% I = 1./gamrnd(a,b,nsamps,1);
% % are the intervales longer or shorter than we need and are any of them
% % within the censored region [0,minINT] or [maxINT,Inf].
% 
% % there's probably a less iterative solution to preserving the min max ints
% % though (like selecting the intervals outside of those bounds and only
% % replacing those)
% while sum(I)>diff(secs)|| sum(I)<diff(secs)-buff || min(I)<minInt || max(I)>maxInt
%     % I = 1./gamrnd(a,b,nsamps,1); ii = ii +1;
%     % %NOTE: this won't necessarily address the overall interval length
%     %conditions being violated directly but hopefully due to randomness
%     %will eventually converge
%     I(I<minInt|I>maxInt)=[];
%     nmissing=nsamps-numel(I);
%     if nmissing==0
%         I=1./gamrnd(a,b,nsamps,1);ii=ii+1;
%     else
%         I = [I; 1./gamrnd(a,b,nmissing,1)]; ii = ii +1;
%     end
% 
% 
%     if ~mod(ii,Icrit)
%         minInt = minInt-0.01;
%         fprintf('%d new minInt: %0.3f',ii/Icrit,minInt)
% 
%     end
% end
% 
% % Create the timepoints from intervals and set the segment onset
% I =  [0; cumsum(I)]+secs(1);
% % Fix the last value to be perfect
% I(end) = secs(2);
% 
% end
% 
% function [I,ii] = unifRandInterval(nsamps,secs)
% 
% arguments
%     nsamps (1,1) double
%     secs (2,1) double
% end
% % disp('hi')
% buff = 0.04;
% % per Poeppel & Assaneo 2-8Hz range consistent across languages/speakers
% minInt = 0.125;
% maxInt = 0.5;
% Icrit = 100000;
% 
% % Original distribution
% % a=6.3;
% % b=1.05;
% 
% % broader distribution
% % a = 3.7;
% % b = 2;
% 
% ii = 0;
% I = 1./unifrnd(1/maxInt,1/minInt,nsamps,1);
% % are the intervales longer or shorter than we need and are any of them
% % within the censored region [0,minINT] or [maxINT,Inf].
% while sum(I)>diff(secs)|| sum(I)<diff(secs)-buff || min(I)<minInt || max(I)>maxInt
%     %NOTE: this should not happens and ii will stay 0 if using unifrnd
%     %instead of gmrnd
%     I = 1./unifrnd(1/maxInt,1/minInt,nsamps,1); ii = ii +1;
% 
%     if ~mod(ii,Icrit)
%         minInt = minInt-0.01;
%         fprintf('%d new minInt: %0.3f',ii/Icrit,minInt)
% 
%     end
% end
% 
% % Create the timepoints from intervals and set the segment onset
% I =  [0; cumsum(I)]+secs(1);
% % Fix the last value to be perfect
% I(end) = secs(2);
% 
% end
