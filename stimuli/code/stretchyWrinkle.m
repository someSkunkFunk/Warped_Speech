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
    'hard_cutoff_hz',8, ...,
    'env_derivative_noise_tol',0, ...
    'min_pkrt_height',0, ...
    'area_thresh',0, ...
    'env_lpf',10, ...
    'rng',0, ...
    'manual_filter',0 ...
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

    otherwise
        error('need to specify which envelope to use.')

end
if strcmp(env_method,'oganian')
    %todo: define env_thresh (if needed?)
    % peakrate -> Ifrom & peakRate struct we're familiar with
    [pkVals,pkTimes,w,p] = findpeaks(diff_env',fs,'MinPeakHeight',warp_config.min_pkrt_height);
    peakRate=struct('pkVals',pkVals,'pkTimes',pkTimes,'p',p,'w',w);
else
    % note: get_peakrate lowpasses the envelope at 10 hz 
    % - keeping to visualize
    [peakRate,env,diff_env,env_thresh]=get_peakRate(env,fs,warp_config);
    %or dont when happy with stimuli output:
    % [peakRate,~,~,env_thresh]=get_peakRate(env,fs,env_thresh_std);
    warp_config.env_thresh=env_thresh;
end
Ifrom=peakRate.pkTimes;p=peakRate.p;w=peakRate.w;
% threshold peaks
Ifrom=Ifrom(p>p_t&w>w_t&(p.*w)>area_t);
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
%inter-segment interval
ISI=0;


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
    slow=1./IPI0_seg<f_center;
    fast=(1./IPI0_seg>f_center);
    % sanity-check + workaround for backwards compatability of rules that
    % affect fast vs slow rates differently:
    too_fast=(1./IPI0_seg)>hard_cutoff_hz;
    if any(too_fast)
        error('there should be no too_fast rates now...')
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
        % RULE 7
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
        % RULE 8
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
        % SAME as rule 7 but less stupid (ie don't worry about being
        % symmetric about any bullshit median, just put shit in the range
        % we want and examine the output
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
                min_stretch_rate=1/0.55;
                % added some wiggle room to max rate 
                % because distinct syllables get 
                % grouped together due to imperpfect peakrate
                % correspondence
                max_stretch_rate=8;
                % % leave overly fast intervals unchanged
                % IPI1_seg(too_fast)=IPI0_seg(too_fast);
                IPI1_seg(~too_fast)=1./(min_stretch_rate+(max_stretch_rate-min_stretch_rate).*rand(sum(~too_fast),1));
        end

    
    end
    
    
    if ss>1
        ISI=Ifrom(seg(ss,1))-Ifrom(seg(ss,1)-1);
        start_t=Ito(end)+ISI;
    else
        %won't ISI just be zero for first segment....?
        start_t=Ifrom(seg(ss,1))+ISI;
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
    if any(isnan(Ito))
        error('wtf duude.')
    end
    % IPF1(seg(ss,1):seg(ss,2))=IPF1_seg;
    % invert segment rates back to time intervals
    % IPI1_seg=1./IPF1_seg;
    
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
S=struct('s',s,'warp_config',warp_config,'wsola_param', ...
    wsola_param,'peakRate',peakRate);
% run these in debug mode:
% inspect_segs(wf,fs,Ifrom,seg,env,p_t,w_t,env_onsets)
% inspect_envelope_derivative(env_onsets,peakRate,fs,warp_config)
% inspect_anchorpoints(wf,wf_warp,fs,s)
end
%% helpers
function [Ifrom, removed_pks]=manually_pick_peaks(wf,fs,Ifrom)
    removed_pks=nan(length(Ifrom),1);
    t=0:1/fs:(length(wf)-1)/fs;
    % define segments to scan thru
    t_seg=6; % in seconds
    len_seg=round(fs*t_seg); % segment length in samples
    n_overlap=round(0.1*len_seg); % number of samples to overlap segments by
    n_offset=len_seg-n_overlap; % amount of samples to slide for next segment 
    n_segments=1+floor((length(wf)-len_seg)/n_offset);
    % preallocate segment indices
    seg_idxs=nan(n_segments,len_seg);
    
    pause_buff=0.5;
    for ss=1:n_segments
        fprintf('segment %d/%d...\n',ss,n_segments)

        seg_idxs(ss,:)=1+(ss-1)*n_offset:(ss-1)*n_offset+len_seg;
        wf_slice=wf(seg_idxs(ss,:));
        t_slice=t(seg_idxs(ss,:));

        Ifrom_slice=Ifrom(Ifrom>t_slice(1)&Ifrom<t_slice(end));
        pk_ys=ones(2,numel(Ifrom_slice));
        pk_ys(2,:)=-1;
        % show waveform + existing peaks for that segment
        figure,plot(t_slice,wf_slice);
        hold on
        plot(repmat(Ifrom_slice',2,1),pk_ys,'color','m')
        ylim([min(wf) max(wf)]);
        xlabel('Time (s)')

        % play audio segment
        soundsc(wf(seg_idxs(ss,:)),fs)
        pause(t_seg+pause_buff)
        % get user input for time windows containing peaks to remove
        rm_windows=validate_input()';

        % find peak within window and remove from Ifrom
        if ~isempty(rm_windows)
            for rr=1:size(rm_windows,1)
                onset=rm_windows(rr,1);
                offset=rm_windows(rr,2);
                % note: assumes a single peak contained within each window,
                % which might not be the case...
                rm_peak_time=Ifrom(Ifrom>onset&Ifrom<offset);
                if numel(rm_peak_time)>1
                    fprintf(['warning, multiple peaks contained in ' ...
                        'window: %0.3f-%0.3f s...\n'],onset,offset)
                end
                Ifrom=Ifrom(Ifrom~=rm_pk_time);
            end
        end
        % regenerate waveform plot with new peaks
        
        % note: we could add information about which peaks were manually
        % removed for peakRate distribution plots... or we could use
        % warp_config to find if peaks that are in peakrate but NOT in s-mat
        % can be attributed to manual filter, and if so just remove them

    
    end
    % save resulting Ifrom...? so we don't have to do manual process twice?
    %probably okay to just do it 

    %what if the peaks we choose yield unsatisfactory results? we won't
    %really know which should have stayed until after we warp the thing...
    %so it may be beneficial to save the Ifrom after all....?

    % aside from that, we may want to add peaks back after removing them
    % and realizing a different peak should be removed instead (by accident
    % or whatever)... add option to re-instate such peaks...

    % remove nans from removed peaks
    removed_pks=removed_pks(~isnan(removed_pks));
    function windows = validate_input(windows)
    %VALIDATE_INPUT  Validate time window input.
    %   WINDOWS = VALIDATE_INPUT(WINDOWS) checks that WINDOWS is either:
    %     (1) a row vector with an even number of elements, representing
    %         [onset1 offset1 onset2 offset2 ...], or
    %     (2) a 2-by-N array, where each column is [onset; offset].
    %   Additional checks:
    %     - Each onset must be strictly less than its corresponding offset.
    %     - Windows must not overlap.
    %   If WINDOWS is invalid, the user will be prompted to re-enter input
    %   until a valid set of windows (or [] for none) is given.
    
        while ~is_valid(windows)
            disp('Invalid input. Please enter windows again.');
            disp('Valid formats:');
            disp(' - Row vector with even length: [on1 off1 on2 off2 ...]');
            disp(' - 2-by-N array: [on1 on2 ...; off1 off2 ...]');
            disp(' - Empty array [] is also valid.');
            prompt='Enter (array of) time window(s) enclosing peak(s) to remove (empty array if none)';
            windows = input(prompt);
        end
        
        % Normalize format: always return 2-by-N
        if isempty(windows)
            return;
        elseif isvector(windows)
            windows = reshape(windows, 2, []);
        end
    end

    function ok = is_valid(w)
        ok = false;
        
        if isempty(w)
            ok = true;
            return;
        end
        
        % Case 1: row vector with even number of elements
        if isvector(w) && size(w,1) == 1 && mod(numel(w),2) == 0
            w = reshape(w, 2, []);
        % Case 2: 2-by-N matrix
        elseif ismatrix(w) && size(w,1) == 2
            % already fine
        else
            return; % invalid shape
        end
        
        onsets = w(1,:);
        offsets = w(2,:);
        
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

