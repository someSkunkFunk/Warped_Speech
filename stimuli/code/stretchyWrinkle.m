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
    center (1,1) double = 0; 
    rule (1,1) = 7;    
    shift_rate (1,1) double = 1.00; 
    peak_tol (1,1) double = 0.0;
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
    % min_mod_interval (1,1) double = 1/4.927;
    % % max_mod_interval (1,1) double = 1/4.32;
    % %trying something a bit slower (25% quantile) cuz too many slow syllables:
    % % max_mod_interval (1,1) double = 1/3.186;
    % % that wasn't low enough, trying some random number:
    % max_mod_interval (1,1) double = 1/2.0;
    % Quantiles (45%-55%) with hard cutoff at 8 hz (0.105, 2.026 p, w): 3.472, 3.949
    min_mod_interval (1,1) double = 1/3.949;
    max_mod_interval (1,1) double = 1/3.472;
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
            % f_center=4.626;
            % Median with hard cutoff at 8 hz (0.105, 2.026 p, w): 3.714
            f_center=3.714;
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

%TODO: determine if gaussian filtering in bark_env function i took from 
% "pushing the envelope" gives better timing results for peakrate events?

switch env_method
    
    case 1 % use broadband envelope
        env = abs(hilbert(wf));
    case 2 % use bark scale filterbank
        env=bark_env(wf,fs,fs);
    case 3
        env=extractGCEnvelope(wf,fs);

end
% note: get_peakrate lowpasses the envelope at 10 hz
[Ifrom,~,p,w,~]=get_peakRate(env,fs,peak_tol);
% set thresholds
p_t=0.105;
w_t=2.026;
Ifrom=Ifrom(p>p_t&w>w_t);
seg=[[1; find(diff(Ifrom)>sil_tol)+1] [find(diff(Ifrom)>sil_tol); length(Ifrom)]];
%inter-segment interval
ISI=0;


n_segs=size(seg,1);
for ss=1:n_segs 
    % % get rates for current segment
    IPI0_seg=diff(Ifrom(seg(ss,1):seg(ss,2)));
    % get original segment duration for post-warp normalization
    seg_dur_0=sum(IPI0_seg);


    slow=1./IPI0_seg<f_center;
    fast=1./IPI0_seg>f_center;
    IPI1_seg=nan(size(IPI0_seg));
    % median vals will stay median
    IPI1_seg(~(slow|fast))=IPI0_seg(~(slow|fast));
    switch rule
        case 1
            % RULE 1
            % % multiply/divide by fixed ratio of input rate
            
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

                    %NEW: correct intervals by duration-specific factor to
                    %get 64s output RE: we didn't like how this turned out
                    % IPI1_seg=IPI1_seg/corrective_factor;
    
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
    % skipped 6 cuz it seemed stupid
    case 7
        switch k
            case 1
                % reg
                IPI1_seg(slow)=1./(f_center-abs(jitter(1)).*rand(size(IPI0_seg(slow))));
                IPI1_seg(fast)=1./(f_center+abs(jitter(2)).*rand(size(IPI0_seg(fast))));
                IPI1_seg(~(fast|slow))=1./f_center;
            case -1
                %irreg
                % need output syllablerate range about median to be
                % symmetric otherwise the mean will shift, but there's
                % still too many "fast" syllables after p,w filtering...
                % crude solution for now is to just leave those out of the
                % "warp"
                peakRate_cutoff=8; % in Hz, rate which is considered too fast to count as new syllable from input distribution

                too_fast=(1./IPI0_seg)>peakRate_cutoff;
                % min_stretch_rate=1/sil_tol;
                min_stretch_rate=.5;
                % mathematically symmetric distance above f_center 
                max_stretch_rate=2*f_center-min_stretch_rate; 
                % leave overly fast intervals unchanged
                IPI1_seg(too_fast)=IPI0_seg(too_fast);
                IPI1_seg(~too_fast)=1./(min_stretch_rate+(max_stretch_rate-min_stretch_rate).*rand(sum(~too_fast),1));
        end
    case 8
        switch k
            case 1
                % reg
                IPI1_seg(slow)=1./(f_center-abs(jitter(1)).*rand(size(IPI0_seg(slow))));
                IPI1_seg(fast)=1./(f_center+abs(jitter(2)).*rand(size(IPI0_seg(fast))));
                IPI1_seg(~(fast|slow))=1./f_center;
            case -1
                % using mean of entire peakrate distribution when
                % thresholding by prominence and peakwidth is done
                % mu=0.2624; %  note: mu is mean in exprnd... but in actual 
                % % exponential distribution function the parameter is lambda, 
                % % whose mean is one over lambda
                lam=1./.2624;
                min_exp_interval=0.0382; % about 26 Hz
                max_exp_interval=sil_tol;
                % IPI1_seg=exprand(mu,size(IPI0_seg));
                % generate uniformly distributed samples
                IPI1_seg=rand(size(IPI0_seg));
                % use inverted, truncated exponential cdf to generate
                % warped samples
                Fmin=1-exp(-lam*min_exp_interval);
                Fmax=1-exp(-lam*max_exp_interval);
                IPI1_seg=-log(1-(Fmin+IPI1_seg.*(Fmax-Fmin)))./lam;

        end

    
    end
    
    
    if ss>1
        ISI=Ifrom(seg(ss,1))-Ifrom(seg(ss,1)-1);
        start_t=Ito(end)+ISI;
    else
        %won't ISI just be zero for first segment....?
        start_t=Ifrom(seg(ss,1))+ISI;
    end
    
    seg_dur_1=sum(IPI1_seg);
    IPI1_seg=IPI1_seg.*(seg_dur_0/seg_dur_1);
    
    % enforce maximum interval/minimum freq allowed in
    % output (note: do after duration normalization to avoid overly-short
    % intervals by accidente)
    IPI1_seg=min(IPI1_seg,interval_ceil_out);
    
    

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
s = [ones(1,2); s; ones(1,2)*length(wf)];
% fix last to match length of silence in original recording's ending
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


