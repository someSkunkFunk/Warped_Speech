function [wf_warp,S] = stretchyWrinkle(wf,fs,warp_config)


%[wf_warp,s] = stretchyWrinkle(wf,fs,k,center,rule,shift_rate,peak_tol,sil_tol,minInt,maxInt,env_method,jitter)
% k: 1->make regular, -1-> more irregular
% center: 1 -> mean, 2 -> median, 3 -> mode
% rule: 
% 1 -> div/multiply freq to stretch/compress      
% 2->add/subtract fixed amt from freq to stretch/compress
% 3-> rerflect about/map to median to "stretch" compress
if nargin < 2
    error('stretchyWrinkle requires at least wf and fs as inputs.');
end

if nargin < 3 || isempty(warp_config)
    warp_config = struct();
end

if ~isstruct(warp_config)
    error('warp_config must be a struct (or omitted).');
end

% defaults 
defaults = struct( ...
    'k', 1, ...
    'center', 0, ...
    'rule', 10, ...
    'shift_rate', 1.00, ...
    'peak_tol', 0.0, ...
    'sil_tol', 0.75, ...
    'min_mod_interval', 1/3.949, ...
    'max_mod_interval', 1/3.472, ...
    'env_method', 2, ...
    'jitter', [0.5, 0.5], ...
    'interval_ceil_out', inf, ...
    'normalize_segments',false,...
    'prom_thresh',0, ... 
    'width_thresh',0, ...
    'env_thresh_std',0, ...
    'hard_cutoff_hz',inf, ...,
    'env_derivative_noise_tol',0, ...
    'min_pkrt_height',0, ...
    'area_thresh',0, ...
    'env_lpf',10, ...
    'rng',0, ...
    'manual_filter',0, ...
    'wav_fnm','none_assigned',...
    'min_stretch_rate',1, ...
    'max_stretch_rate',10, ...
    'elongation_thresh',inf,...
    'shortening_thresh',0 ...
    );

% copy missing fields from defaults into warp_config
flds = fieldnames(defaults);
for iF = 1:numel(flds)
    f = flds{iF};
    if ~isfield(warp_config,f) || isempty(warp_config.(f))
        warp_config.(f) = defaults.(f);
    end
end
% unpack warp_config into local vars for readability
k               = warp_config.k;
center          = warp_config.center;
rule_num            = warp_config.rule_num;
shift_rate      = warp_config.shift_rate;
% note: i think this was originally the prominence threshold and thus
% redundant... but maybe still useful as an absolute peak theshold
peak_tol        = warp_config.peak_tol;
sil_tol         = warp_config.sil_tol;
min_mod_interval= warp_config.min_mod_interval;
max_mod_interval= warp_config.max_mod_interval;
env_method      = warp_config.env_method;
jitter          = warp_config.jitter;
interval_ceil_out=warp_config.interval_ceil_out;
normalize_segments=warp_config.normalize_segments;
p_t=warp_config.prom_thresh;
w_t=warp_config.width_thresh;
area_t=warp_config.area_thresh;
% how many std below which to zero-out envelope - note that zero is
% equivalent with half-wave rectification
env_thresh_std=warp_config.env_thresh_std;
hard_cutoff_hz=warp_config.hard_cutoff_hz;
env_derivative_noise_tol=warp_config.env_derivative_noise_tol;
min_stretch_rate=warp_config.min_stretch_rate;
max_stretch_rate=warp_config.max_stretch_rate;
% max (warp_interval/original_interval) ratio allowed for output
elongation_thresh=warp_config.elongation_thresh;
shortening_thresh=warp_config.shortening_thresh;
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

switch warp_config.rng
    case 1
        % For repeatability
        rng(1);
    case 0
    otherwise
        error('not recognized.')
        
end


% Extract envelope

%TODO: determine if gaussian filtering in bark_env function i took from 
% "pushing the envelope" gives better timing results for peakrate events?

switch env_method
    
    case 'hilbert' % use broadband envelope
        env = abs(hilbert(wf));
    case 'bark' % use bark scale filterbank - scraped from oganian peakrate function
        env=bark_env(wf,fs,fs);
    case 'gammaChirp'
        env=extractGCEnvelope(struct('wf',wf,'fs',fs),fs);
    case 'oganian'
        % fully rely on all the peakrate algorithm steps in oganian method
        [env, ~, ~,diff_env]=find_peakRate_oganian(wf, fs, [], 'loudness', fs);
        % note: oganian function does this already when evaluating peakrate 
        % but gives entire sound series + we want to be able to filter peaks
        % afterwards
        diff_env(diff_env<0)=0;
    case 'textgrid'
        % doesn't extract envelope at all, just uses textgrids to locate
        % syllables directly (using WebMAUS 'Pipeline name' -
        % 'G2P->MAUS->PHO2SYL')

    otherwise
        error('need to specify which envelope to use.')

end
switch env_method
    case 'oganian'
        %todo: define env_thresh (if needed?)
        % peakrate -> Ifrom & peakRate struct we're familiar with
        [pkVals,pkTimes,w,p] = findpeaks(diff_env',fs,'MinPeakHeight',warp_config.min_pkrt_height);
        peakRate=struct('pkVals',pkVals,'pkTimes',pkTimes,'p',p,'w',w);
    case {'hilbert','bark','gammaChirp'}
        % note: get_peakrate lowpasses the envelope at 10 hz 
        % - keeping to visualize
        [peakRate,env,diff_env,env_thresh]=get_peakRate(env,fs,warp_config);
        %or dont when happy with stimuli output:
        % [peakRate,~,~,env_thresh]=get_peakRate(env,fs,env_thresh_std);
        warp_config.env_thresh=env_thresh;
    case 'textgrid'
end

switch env_method
    case 'textgrid'
        % use syllable times from webmaus forced-aligner
        % todo: read in file for current stimulus into function
        Ifrom=read_syll_from_textgrid(warp_config.wav_fnm);
        
    case {'oganian','hilbert','bark','gammaChirp'}
        % use acoustic info from peak rate to index syllables
        Ifrom=peakRate.pkTimes;p=peakRate.p;w=peakRate.w;
        % threshold peaks
        Ifrom=Ifrom(p>p_t&w>w_t&(p.*w)>area_t);
end
% recursively remove rates that are too fast - recalculate with filtered
        % times
if any(diff(Ifrom<hard_cutoff_hz))
    % make a dummy mask to make recursive function compatible with how we used
    % it in optimize_peakRate_algo
    fprintf('applying recursive cutoff at %0.3f Hz...\n', hard_cutoff_hz)
    ff=true(length(Ifrom),1);
    Ifrom=Ifrom(recursive_cutoff_filter(ff,Ifrom,hard_cutoff_hz));
end
% manual peak removal
if warp_config.manual_filter
    [Ifrom, warp_config.manually_removed_pks]=manually_pick_peaks(wf,fs,Ifrom);
end
seg=[[1; find(diff(Ifrom)>sil_tol)+1] [find(diff(Ifrom)>sil_tol); length(Ifrom)]];
% preallocate Ito
Ito=nan(size(Ifrom));
%inter-segment interval
% ISI=0;


n_segs=size(seg,1);
for ss=1:n_segs 
    % % get rates for current segment
    IPI0_seg=diff(Ifrom(seg(ss,1):seg(ss,2)));
    % get original segment duration for post-warp normalization
    seg_dur_0=sum(IPI0_seg);
    IPI1_seg=nan(size(IPI0_seg));
    % note: recursive cutoff removes need to apply stuff below here
    % filter_fast_intervals=true;
    % if filter_fast_intervals
    %     too_fast=(1./IPI0_seg)>peakRate_cutoff;
    % else
    %     too_fast=(1./IPI0_seg)>inf;
    % end
    % % leave overly fast intervals unchanged
    % IPI1_seg(too_fast)=IPI0_seg(too_fast);
    % slow=1./IPI0_seg<f_center;
    % fast=(1./IPI0_seg>f_center)&~too_fast;
    slow=(1./IPI0_seg)<f_center;
    fast=(1./IPI0_seg)>f_center;
    % sanity-check + workaround for backwards compatability of rules that
    % affect fast vs slow rates differently:
    too_fast=(1./IPI0_seg)>hard_cutoff_hz;
    if any(too_fast) && ~strcmp(env_method,'textgrid')
        % turns out the reader routinely says syllables at rates higher
        % than 10 Hz with unknown upper bound (probably 12-14 hz)
        warning('this caused weird error before proceed with caution...')
    end
    
    switch rule_num
        case 1
            % RULE 1
            % % multiply/divide by fixed ratio of input rate
            
            rate_shift=(1+abs(shift_rate));

            switch k
                case -1
                    % reg -> shift towards center f
                    % IPF1_seg(slow)=IPF0_seg(slow).*rate_shift;
                    % IPF1_seg(fast)=IPF0_seg(fast)./rate_shift;

                    % this is equivalent to above when working in 
                    % time domain cuz reciprocals:
                    IPI1_seg(slow)=IPI0_seg(slow)./rate_shift;
                    IPI1_seg(fast)=IPI0_seg(fast).*rate_shift;

                    % this keeps percent change constant but probably not what we want
                    % IPF1(fast)=IPF0(fast).*(1-amt_shift);
                case 1
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
                case -1
                    % reg -> shift towards center f
                    % slow intervals get faster (shorter)
                    IPI1_seg(slow)=IPI0_seg(slow)./rate_shift(slow);
                    % fast intervals get slower (longer)
                    IPI1_seg(fast)=IPI0_seg(fast).*rate_shift(fast);
                case 1
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
            case -1
                % reg -> shift towards center f
                IPI1_seg(slow)=1./(f_center+jitter.*(2.*rand(size(IPI0_seg(slow)))-1));
                IPI1_seg(fast)=1./(f_center+jitter.*(2.*rand(size(IPI0_seg(fast)))-1));
            case 1
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
            case -1
                % reg -> shift towards center f
                IPI1_seg(slow)=1./(f_center+jitter.*(2.*rand(size(IPI0_seg(slow)))-1));
                IPI1_seg(fast)=1./(f_center+jitter.*(2.*rand(size(IPI0_seg(fast)))-1));
            case 1
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
            case -1
                % reg
                IPI1_seg(slow)=1./(f_center-abs(jitter(1)).*rand(size(IPI0_seg(slow))));
                IPI1_seg(fast)=1./(f_center+abs(jitter(2)).*rand(size(IPI0_seg(fast))));
                IPI1_seg(~(fast|slow))=1./f_center;
            case 1
                % irreg -> fast go to fast maximally and slow go to slow
                % maximally
                IPI1_seg(slow)=1./(1/max_mod_interval+jitter.*(rand(size(IPI0_seg(slow)))));
                IPI1_seg(fast)=1./(1/min_mod_interval-jitter.*(rand(size(IPI0_seg(fast)))));
        end
    % skipped 6 cuz it seemed stupid
    case 7
        % RULE 7 map to uniform RATE distribution
        % close to f-center in reg and with 
        % wide lims in irreg
        switch k
            case -1
                % reg
                IPI1_seg(slow)=1./(f_center-abs(jitter(1)).*rand(size(IPI0_seg(slow))));
                IPI1_seg(fast)=1./(f_center+abs(jitter(2)).*rand(size(IPI0_seg(fast))));
                IPI1_seg(~(fast|slow))=1./f_center;
            case 1
                %irreg
                % need output syllablerate range about median to be
                % symmetric otherwise the mean will shift, but there's
                % still too many "fast" syllables after p,w filtering...
                % crude solution for now is to just leave those out of the
                % "warp"
                % peakRate_cutoff=8; % in Hz, rate which is considered too fast to count as new syllable from input distribution

                % too_fast=(1./IPI0_seg)>peakRate_cutoff;
                % min_stretch_rate=1/sil_tol;
                min_stretch_rate=.5;
                % mathematically symmetric distance above f_center 
                % NOTE: not longer mathematically symmetric because this is
                % not the TRUE f_center if we're ignoring stuff above 8 Hz
                max_stretch_rate=2*f_center-min_stretch_rate; 
                % leave overly fast intervals unchanged
                IPI1_seg(too_fast)=IPI0_seg(too_fast);
                IPI1_seg(~too_fast)=1./(min_stretch_rate+(max_stretch_rate-min_stretch_rate).*rand(sum(~too_fast),1));
        end
    case 8
        % RULE 8 map to narrow RATE distribution in reg & use logrand
        % INTERVAL distribution for irreg
        switch k
            case -1
                % reg
                IPI1_seg(slow)=1./(f_center-abs(jitter(1)).*rand(size(IPI0_seg(slow))));
                IPI1_seg(fast)=1./(f_center+abs(jitter(2)).*rand(size(IPI0_seg(fast))));
                IPI1_seg(~(fast|slow))=1./f_center;
            case 1
                % using mean of entire peakrate distribution when
                % thresholding by prominence and peakwidth is done
                % mu=0.2624; %  note: mu is mean in exprnd... but in actual 
                % % exponential distribution function the parameter is lambda, 
                % % whose mean is one over lambda
                % corrective factor based on ratio difference of durations
                % from poisson with 12 Hz max, og mean for lambda (relative
                % to og durations)
                % mean_duration_correction_factor=1.0592;
                %increased interval after noticing fast intervals are
                %unintelligible -> lowered max to 10 Hz + outputs still too
                %long (just educated guess)
                mean_duration_correction_factor=1.65;
                mean_interval=.2624;
                lam=1./(mean_interval/mean_duration_correction_factor);
                % min_exp_interval=0.0382; % about 26 Hz
                % min_exp_interval=0.0382*2; % while not many samples near upper limit generated randomly, the duration normalization seemed to speed up a lot of intervals excessively
                min_exp_interval=1/10; % fast intervals when max is 12 (above) are really hard to comprehend...
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
        case 9
        % RULE 9
        % reg: map to narrow RATE distribution
        % irreg: map to wide uniform INTERVAL distribution
        % SIMILAR to rule 7 but for INTERVALS in irreg
        switch k
            case -1
                % reg
                IPI1_seg(slow)=1./(f_center-abs(jitter(1)).*rand(size(IPI0_seg(slow))));
                IPI1_seg(fast)=1./(f_center+abs(jitter(2)).*rand(size(IPI0_seg(fast))));
                IPI1_seg(~(fast|slow))=1./f_center;
            case 1
                %irreg
                % need output syllablerate range about median to be
                % symmetric otherwise the mean will shift, but there's
                % still too many "fast" syllables after p,w filtering...
                % crude solution for now is to just leave those out of the
                % "warp"
                % peakRate_cutoff=8; % in Hz, rate which is considered too fast to count as new syllable from input distribution

                % too_fast=(1./IPI0_seg)>peakRate_cutoff;
                max_stretch_interval=sil_tol;
                min_stretch_interval=1/peakRate_cutoff;
                % leave overly fast intervals unchanged
                IPI1_seg(too_fast)=IPI0_seg(too_fast);
                IPI1_seg(~too_fast)=(min_stretch_interval+(max_stretch_interval-min_stretch_interval).*rand(sum(~too_fast),1));
        end
        case 10
        % RULE 10
        % SAME as rule 9 but we use uniform distribution in rates rather
        % than intervals
        switch k
            % actually it looks to me like reg case here still warps the
            % RATE distribution not the intervals directly...
            case -1
                % reg
                % IPI1_seg(slow)=1./(f_center-abs(jitter(1)).*rand(size(IPI0_seg(slow))));
                % IPI1_seg(fast)=1./(f_center+abs(jitter(2)).*rand(size(IPI0_seg(fast))));
                % IPI1_seg(~(fast|slow))=1./f_center;
                % i think keeping fast stuff slightly faster and slow stuff
                % slightly slower might benefit overall quality in terms of
                % minimizing warp artefacts, but we aren't concerned with
                % this in the irreg case and we also don't want to retain
                % stress-syllable timing cues so better to just assing
                % intervals randomly across entire range
                min_compress_rate=f_center-abs(jitter(1));
                max_compress_rate=f_center+abs(jitter(2));
                IPI1_seg(~too_fast)=1./(min_compress_rate+(max_compress_rate-min_compress_rate).*rand(sum(~too_fast),1));
            case 1
                %irreg
                % generate random rates from uniform distribution across
                % range of possible values 
                min_stretch_rate=1./sil_tol;
                max_stretch_rate=hard_cutoff_hz;
                % % leave overly fast intervals unchanged
                % IPI1_seg(too_fast)=IPI0_seg(too_fast);
                IPI1_seg(~too_fast)=1./(min_stretch_rate+(max_stretch_rate-min_stretch_rate).*rand(sum(~too_fast),1));
        end

        case 11
        % RULE 11
        % SAME as rule 10 but we use logrnd distribution in reg case rather
        % than uniform
        % and maybe we normalize the irreg case by raising the minimum
        % stretch rate rather than normalizing (once we figure out how long
        % they come out to be)
        switch k
            
            case -1
                % reg
                % generate random samples from log-normal distribution
                % note the parameters of lognrnd are the mean and std of
                % the associated normal distribution, rather than the
                % lognormal distribution itself so we have to
                % reverse-engineer mu and sigma from the desired lognormal 
                % mean and
                % variance
                M=f_center;
                V=jitter(1);
                mu=log(M^2/sqrt(V+M^2));
                sigma=sqrt(log(V/M^2+1));
                IPI1_seg(~too_fast)=1./lognrnd(mu,sigma,sum(~too_fast),1);
                    
            case 1 
                %irreg

                % generate random rates from uniform distribution across
                % range of possible values plus a slightly higher lower
                % bound

                % min stretch rate mostly drove the duration discrepancies,
                % so tuned it such that the durations are not too far from
                % original
            
                % note: everything below until clear statement could be a
                % function
                ok_=false;
                maxiter_=1e6;
                ii_=0;
                IPI1_seg(~too_fast)=get_rand_rate_intervals(min_stretch_rate,max_stretch_rate,sum(~too_fast));
                while (~ok_)&&(ii_<maxiter_)
                    ii_=ii_+1;
                    %note: "too fast" should be ok automatically since not
                    %warped at all.... so long as elongation/shortening
                    %thresh create interval that contains 1
                    elongation_ok_mask_=(IPI1_seg./IPI0_seg)<elongation_thresh;
                    shortening_ok_mask_=(IPI1_seg./IPI0_seg)>shortening_thresh;
                    if all(elongation_ok_mask_)&&all(shortening_ok_mask_)
                        ok_=true;
                    else
                        % only resample out-of-bounds intervals
                        IPI1_seg(~elongation_ok_mask_|~shortening_ok_mask_)=get_rand_rate_intervals(min_stretch_rate,max_stretch_rate,sum(~elongation_ok_mask_|~shortening_ok_mask_));
                    end
                end
                if ii_>=maxiter_
                    warning('iter threshold surpassed.')
                else
                    disp('n iterations to find valid set:')
                    disp(ii_)
                end
                clear ii_ ok_ maxiter_ elongation_ok_mask_ shortening_ok_mask_
        end
        case 12
            % RULE 12
            % reg: map all INTERVALS to single value 
            % irreg: map below-median INTERVALS to bounded submedian uniform
            % note: already computed fast/slow before switch but I don't
            % trust it and don't wanna worry about how it affects preceding
            % rules atm so just re-evaluating in time domain

            fast=IPI0_seg>(1/f_center);
            slow=IPI0_seg<(1/f_center);
            switch k
                case -1
                    %reg
                    IPI1_seg(~too_fast)=1/f_center;
                case 1
                    %irreg
                    min_interval=1/max_stretch_rate;
                    max_interval=1/min_stretch_rate;
                    IPI1_seg(slow)=get_rand_intervals(min_interval,1/f_center,sum(slow));
                    IPI1_seg(fast)=get_rand_intervals(1/f_center,max_interval,sum(fast));
            end
        case 13
            % RULE 13
            % same as 12 but distributes at-median values between the two
            % output distributions using a biased coin flip
            % reg: map all INTERVALS to single value 
            % irreg: map below-median INTERVALS to bounded submedian uniform
            % note: already computed fast/slow before switch but I don't
            % trust it and don't wanna worry about how it affects preceding
            % rules atm so just re-evaluating in time domain

            fast=IPI0_seg>(1/f_center);
            slow=IPI0_seg<(1/f_center);
            switch k
                case -1
                    %reg
                    IPI1_seg(~too_fast)=1/f_center;
                case 1
                    %irreg
                    min_interval=1/max_stretch_rate;
                    max_interval=1/min_stretch_rate;
                    IPI1_seg(slow)=get_rand_intervals(min_interval,1/f_center,sum(slow));
                    IPI1_seg(fast)=get_rand_intervals(1/f_center,max_interval,sum(fast));
                    if any(IPI1_seg(~(fast|slow)))
                        % "bias" determined from empirical distribution
                        % (based on textgrid times here)
                        % goal is to balance the output distribution by
                        % asymetrically assigning at-median values to
                        % output distributions
                        disp('tie-breaking with biased flip')
                        flips_=get_biased_flips(sum(~(fast|slow)));
                        ipis1_=nan(n_flips,1);
                        ipis1_(flips_)=get_rand_intervals(1/f_center,max_interval,sum(flips_));
                        ipis1_(~flips_)=get_rand_intervals(min_interval,1/f_center,sum(~flips_));
                        IPI1_seg(~(fast|slow))=ipis1_;
                        clear flips_ ipis1_
                    end     
            end
        case 14
            % rule 14: make flat uniform irreg based on two uniform 
            % distributions (for vals below/above median) determined by
            % extrema
            slow=IPI0_seg<(1/f_center);
            fast=IPI0_seg>(1/f_center);
            switch k
                case -1
                    %reg
                    IPI1_seg(~too_fast)=1/f_center;
                case 1
                    %irreg
                    min_interval=(1/max_stretch_rate);
                    max_interval=(1/min_stretch_rate);
                    mid_interval=mean([min_interval max_interval]);
                    IPI1_seg(slow)=get_rand_intervals(min_interval, ...
                        mid_interval,sum(slow));
                    IPI1_seg(fast)=get_rand_intervals(mid_interval, ...
                        max_interval,sum(fast));
                    if any(IPI1_seg(~(fast|slow)))
                        % "bias" determined from empirical distribution
                        % (based on textgrid times here)
                        % goal is to balance the output distribution by
                        % asymetrically assigning at-median values to
                        % output distributions
                        disp('tie-breaking with biased flip')
                        flips_=get_biased_flips(sum(~(fast|slow)));
                        ipis1_=nan(n_flips,1);
                        ipis1_(flips_)=get_rand_intervals(mid_interval, ...
                            max_interval,sum(flips_));
                        ipis1_(~flips_)=get_rand_intervals(min_interval, ...
                            mid_interval,sum(~flips_));
                        IPI1_seg(~(fast|slow))=ipis1_;
                        clear flips_ ipis1_
                    end    
            end
    
    end
    if any(isnan(IPI1_seg))
        % not sure how rule 12 handled median values in input... but some
        % should have been equal apparently, and thus remained nan...
        error('not all output times assigned.')
    end
    
    

    if ss==1
        start_t=Ifrom(seg(ss,1));
    else
        switch k
            case -1
                %reg
                %round original inter-segment interval down to multiple
                %of 1/f_center
                ISI=Ifrom(seg(ss,1))-Ifrom(seg(ss-1,2));
                ISI=(1/(f_center))*floor(ISI*f_center);
            case 1
                %irreg
                % get original inter-segment interval
                ISI=Ifrom(seg(ss,1))-Ifrom(seg(ss-1,2));
        end
        % start time at end of previous Ito segment
        start_t=Ito(seg(ss-1,2))+ISI;
    end
    if normalize_segments
        seg_dur_1=sum(IPI1_seg);
        IPI1_seg=IPI1_seg.*(seg_dur_0/seg_dur_1);
    end
    
    % enforce maximum interval/minimum freq allowed in
    % output (note: do after duration normalization to avoid overly-short
    % intervals by accidente)
    IPI1_seg=min(IPI1_seg,interval_ceil_out);
    
    

    %CUMSUM
    Ito(seg(ss,1):seg(ss,2),1)=[start_t; start_t+cumsum(IPI1_seg)];
    
    % IPF1(seg(ss,1):seg(ss,2))=IPF1_seg;
    % invert segment rates back to time intervals
    % IPI1_seg=1./IPF1_seg;
    
end
if any(isnan(Ito))||~isequal(size(Ifrom),size(Ito))
    error('nans in Ito or size doesnt match Ifrom.')
end

% convert to indices
s = round([Ifrom Ito]*fs);
% pad indices with correct start/end times
s = [ones(1,2); s; ones(1,2)*length(wf)];
% fix last to match length of silence in original recording's ending
end_sil = diff(s(:,1)); end_sil = end_sil(end);
s(end,2) = s(end-1,2)+end_sil;
% warp
wsola_param.tolerance = 256;
wsola_param.synHop = 256;
wsola_param.win = win(1024,2); % hann window
wf_warp = wsolaTSM(wf,s,wsola_param);
if rule_num==11 && k==1
    % janky fix to add this parameter for recordkeeping only once to
    % warp_config
    warp_config.stretch_rate_lims=[min_stretch_rate max_stretch_rate];
end
switch env_method
    case {'hilbert','oganian','gammaChirp','bark'}
        S=struct('s',s,'warp_config',warp_config,'wsola_param', ...
            wsola_param,'peakRate',peakRate);
    case 'textgrid'
        S=struct('s',s,'warp_config',warp_config,'wsola_param', ...
            wsola_param,'syllable_times',Ifrom);
end
% run these in debug mode:
% inspect_segs(wf,fs,Ifrom,seg,env,p_t,w_t,env_onsets)
% inspect_envelope_derivative(env_onsets,peakRate,fs,warp_config)
% inspect_anchorpoints(wf,wf_warp,fs,s)
end
%% helpers
function flips=get_biased_flips(n_flips)
    n_above=18; % number of median vals we want flipping to above
    n_below=277+18; % number of median vals we want to remain below
    N=n_above+n_below; % should be 313
    bias=n_above/N;
    
    probs=rand(n_flips,1);
    flips=false(size(probs));
    flips(probs>bias)=true;
    
end

function rand_intervals=get_rand_intervals(min_interval,max_interval,n_intervals)
% rand_intervals=get_rand_intervals(min_interval,max_interval,n_intervals)
    rand_intervals=min_interval+(max_interval-min_interval).*rand(n_intervals,1);
end

function rand_rate_intervals=get_rand_rate_intervals(min_stretch_rate,max_stretch_rate,n_rates)
% generate random INTERVALS by inverting uniform RATE distribubtion
% sample a uniform RATE distribution between
% min_stretch_rate,max_stretch_rate and taking their reciprocal
% n_rates: number of random samples to generate
    rand_rate_intervals=1./(min_stretch_rate+(max_stretch_rate-min_stretch_rate).*rand(n_rates,1));
end

function syll_times=read_syll_from_textgrid(wav_fnm)
    % reads syllable times from textgrid files
    global boxdir_mine

    % todo: make tgFName variable fed into function - can just pass wav fpth
    % from outer script to here 
    tgFName=fullfile(boxdir_mine,'stimuli','wrinkle','syllable_textgrids',...
    sprintf('%s.TextGrid',wav_fnm));
    % tgFName='C:/Users/ninet/Box/my box/LALOR LAB/oscillations project/MATLAB/Warped Speech/stimuli/wrinkle/syllable_textgrids/wrinkle001.TextGrid';
    % whichTier=[]; % load all tiers
    whichTier='MAS'; % syllable tier only
    [TextGridStruct,~,~,~]= ReadTextGrid(tgFName, whichTier);
    pause_symbol='<p:>';
    % make a mask for syllable onsets
    syll_m=cellfun(@(x) ~strcmp(pause_symbol,x),TextGridStruct.labs);
    % use onsets
    % syll_m=[syll_m, false(size(syll_m))];
    % use offsets
    % syll_m=[false(size(syll_m)),syll_m];
    % get onset/offset times
    % syll_times=TextGridStruct.segs(syll_m);
    syll_times=mean(TextGridStruct.segs(syll_m,:),2);
end

function [Ifrom, removed_pks]=manually_pick_peaks(wf,fs,Ifrom)
    % removed_pks=nan(length(Ifrom),1);
    removed_pks=[];
    % define segments to scan thru
    t_seg=3; % in seconds
    len_seg=round(fs*t_seg); % segment length in samples
    n_overlap=round(0.1*len_seg); % number of samples to overlap segments by
    n_offset=len_seg-n_overlap; % amount of samples to slide for next segment
    % this seems short by one segment...?
    % n_slices=floor(1+(length(wf)-len_seg)/n_offset);
    % pad the signal instead to a value divisible by number of segments
    n_slices=ceil((length(wf)-len_seg)/n_offset) + 1;
    n_pad=((n_slices-1)*n_offset+len_seg)-length(wf);
    % assumes wf is col vector...
    wf=cat(1,wf,zeros(n_pad,1));
    % define t-vec after padding so last segment doesnt cause bug
    t=0:1/fs:(length(wf)-1)/fs;
    % preallocate slice indices... why again?
    slice_idxs=nan(n_slices,len_seg);
    pause_buff=0.5;
    for ss=1:n_slices
        fprintf('segment %d/%d...\n',ss,n_slices)

        slice_idxs(ss,:)=1+(ss-1)*n_offset:(ss-1)*n_offset+len_seg;
        [wf_slice,t_slice]=plot_slice(slice_idxs(ss,:),wf,t,Ifrom);
        title(sprintf('segment %d/%d before\n',ss,n_slices))

        % play audio segment
        soundsc(wf_slice,fs)
        pause(t_seg+pause_buff)
        
        satisfied=false;
        while ~satisfied
            % get user input for time windows containing peaks to remove
            % rm_windows=validate_input_windows();
            % get user input for precise times then generate windows around
            % it
            rm_windows=times_to_windows();
            if isempty(rm_windows)
                % don't need to remove peaks
                disp('no peaks removed.')
                satisfied=true;
            else
                if isequal(rm_windows,0)
                    % replay audio segment
                    subslice=input(['specify time lims to replay ' ...
                        '(0 if whole slice)']);
                    if isequal(subslice,0)
                        soundsc(wf_slice,fs)
                    else
                        try
                            soundsc(wf_slice(t_slice>min(subslice)&t_slice<max(subslice)),fs)
                        catch
                            disp('replay lims specified incorrectly. replaying full slice.')
                            soundsc(wf_slice,fs);
                        end
                    end
                else
                    % find peak within window and remove from Ifrom
                    Ifrom_rm=rm_peaks(Ifrom,rm_windows);
                    % regenerate waveform plot with new peaks
                    [~,~]=plot_slice(slice_idxs(ss,:),wf,t,Ifrom_rm);
                    title(sprintf('segment %d/%d after\n',ss,n_slices))
                    soundsc(wf_slice,fs)
                    pause(t_seg+pause_buff)
                    ok=input('Do peaks look ok now?\n');
                    if ok
                        %codify removed peaks and move on to next segment
                        satisfied=true;
                        % record which peaks where removed
                        rmd_m=~ismember(Ifrom,Ifrom_rm);
                        removed_pks=cat(1,removed_pks,Ifrom(rmd_m));
                        % update Ifrom
                        Ifrom=Ifrom_rm;
                    end
                end
            end
        end
        
        % note: we could add information about which peaks were manually
        % removed for peakRate distribution plots... or we could use
        % warp_config to find if peaks that are in peakrate but NOT in s-mat
        % can be attributed to manual filter, and if so just remove them
    close all
    end
    % save resulting Ifrom...? so we don't have to do manual process twice?
    %probably okay to just do it 

    % remove nans from removed peaks
    % removed_pks=removed_pks(~isnan(removed_pks));

    function Ifrom=rm_peaks(Ifrom,rm_windows)
        npks_pre=numel(Ifrom);
        if ~isempty(rm_windows)
            for rr=1:size(rm_windows,1)
                onset=rm_windows(rr,1);
                offset=rm_windows(rr,2);
                % note: assumes a single peak contained within each window,
                % which might not be the case...
                rm_m=Ifrom>onset&Ifrom<offset;
                if sum(rm_m)>1
                    fprintf(['warning, multiple peaks contained in ' ...
                        'window: %0.3f-%0.3f s...\n'],onset,offset)
                end
                Ifrom(rm_m)=[];
            end
        end
        npks_post=numel(Ifrom);
        fprintf('%d peaks removed...\n',npks_pre-npks_post)
    end
    function rm_windows=times_to_windows()
        % assumes times specified are always positive and at least 0+dt
        % if zero is entered, outputs zero so audio can be replayed
        prompt='enter approximate peak times (to 3 decimal places in seconds) to remove.\n';
        times=input(prompt);
        %note: consider adding an input validator (to avoid having to
        %re-start due to badly-defined inputs
        % assumes row vector of numbers given
        while ~is_valid(times)
            disp('times specified incorrectly - must be row vector or empty array.')
            times=input(prompt);
        end
        if isempty(times)
            rm_windows=[];return
        elseif isequal(times,0)
            rm_windows=0;return
        else
            % half-width of temporal window around specified time - based on 3
            % decimal place rounding precision
            dt=0.001; % in s
            %preallocate output 
            rm_windows=nan(numel(times),2);
            for tt=1:numel(times)   
                rm_windows(tt,:)=[times(tt)-dt, times(tt)+dt];
            end
        end

        function ok=is_valid(times)
            ok=false;
            if isrow(times)||isempty(times)
                ok=true;
            end
        end

    end

    function windows = validate_input_windows()
    %VALIDATE_INPUT  Validate time window input.
    %   WINDOWS = VALIDATE_INPUT(WINDOWS) checks that WINDOWS is either:
    %     (1) a row vector with an even number of elements, representing
    %         [onset1 offset1 onset2 offset2 ...], or
    %     (2) a N-by-2 array, where each row is [onset offset;].
    %   Additional checks:
    %     - Each onset must be strictly less than its corresponding offset.
    %     - Windows must not overlap.
    %   If WINDOWS is invalid, the user will be prompted to re-enter input
    %   until a valid set of windows (or [] for none) is given.
        prompt=['Enter (array of) time window(s) enclosing peak(s)' ...
            '\nto remove (empty array if none):\n'];
        windows = input(prompt);
        while ~is_valid(windows)
            disp('Invalid input. Please enter windows again.');
            disp('Valid formats:');
            disp(' - Row vector with even length: [on1 off1 on2 off2 ...]');
            disp(' - N-by-2 array: [on1 off1 ...; on2 off2 ...]');
            disp(' enter 0 to replay sound slice');
            disp(' - Empty array [] is also valid.');
            windows = input(prompt);
        end

        % Normalize format: always return N-by-2
        if isempty(windows)||isequal(windows,0)
            return;
        elseif isvector(windows)
            windows = reshape(windows, 2, [])';
        end

        function ok = is_valid(w)
            ok = false;

            if w==0
                ok=true;
                return;
            end

            if isempty(w)
                ok = true;
                return;
            end

            % Case 1: row vector with even number of elements
            if isrow(w) && mod(numel(w),2) == 0
                % works but should reshape outside of is_valid
                % w = reshape(w, 2, [])';
            % Case 2: N-by-2 matrix
            elseif ismatrix(w) && size(w,2) == 2
                % already fine
            else
                return; % invalid shape
            end

            onsets = w(:,1);
            offsets = w(:,2);

            % Check onset < offset
            if any(offsets <= onsets)
                return;
            end

            % Check for overlap
            [~, order] = sort(onsets);
            onsets = onsets(order);
            offsets = offsets(order);

            if any(onsets(2:end) < offsets(1:end-1))
                return;
            end

            ok = true;
        end
    end

    function [wf_slice,t_slice]=plot_slice(slice_idx,wf,t,Ifrom)
        wf_slice=wf(slice_idx);
        t_slice=t(slice_idx);
        t_start=t_slice(1);
        t_end=t_slice(end);

        Ifrom_slice=Ifrom(Ifrom>t_slice(1)&Ifrom<t_slice(end));
        pk_ys=ones(2,numel(Ifrom_slice));
        pk_ys(2,:)=-1;
        text_ys=0.6*ones(1,length(Ifrom_slice));
        % flip every other across xaxis for readibility
        text_ys(2:2:numel(Ifrom_slice))=-text_ys(2:2:numel(Ifrom_slice));
        time_lbls=cell(1,numel(Ifrom_slice));
        for ll=1:numel(Ifrom_slice)
            time_lbls{ll}=sprintf('%0.3fs',Ifrom_slice(ll));
            
        end
        % show waveform + existing peaks for that segment
        figure,plot(t_slice,wf_slice);
        hold on
        plot(repmat(Ifrom_slice',2,1),pk_ys,'color','m')
        text(Ifrom_slice', text_ys,time_lbls, ...
            "FontSize",9)
        ylim([min(wf) max(wf)]);
        xlim([t_start t_end]);
        
        % add grid labels to help read times
        dxt=0.1;
        xt=t_start:dxt:t_end;
        xt_labels=strings(size(xt));
        xt_labels(mod(xt,0.5)==0)=string(xt(mod(xt,0.5)==0));
        grid on
        xlabel('Time (s)')
        set(gca,'XTick',xt,'XTickLabel',xt_labels);
    end

end

function y=reflect_about(x,xr)
    y=x-2.*(x-xr);
end

function ff_cleaned=recursive_cutoff_filter(ff,all_times,syllable_cutoff_hz)
    
    ff_rates=1./diff(all_times(ff));
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

