%% TRF Kernel Demo
% Parametric evoked repsonse ("TRF") kernel
% 
% The kernel is a sum of K simple components with interpretable parameters
% (amplitude, onset delay, time constants).
% The script:
%   1. builds the kernel and shows its components, the sum, the peak
%   latencies/amplitudes and the power spectrum
%   2. sweeps ONE parameter and shows how latencies, aplitudes, and
%   spectrum move
%   3. shows how the kernel is time-compressed / stretched with speech
%   rate.

% Component types (each normalized to unit peak, so 'amp' = its stand-alone
% peak):
%   'gamma': (t/n*tau))^n * exp(n-t/tau)            peak at delay + n*tau
%   'biexp': exp(-t/tauDecay) - exp(-t/tauRise) difference of two exponentials
%   'dsine': exp(-t/tau) * sin(2*pi*freq*t) damped oscillator (aka impulse
%   response of a damped linear oscillator, e.g. a Stuart-Landau unit below
%   its Hopf bifurcation)
% Here t is time since component's onset (its delay); all causal.
% 
% quick start: pick cfg.preset, edit the component structs in section 2,
% and run.
% Things to try:
%   1. change tau or delay and watch printed peaks of the SUMMED kernel 
% (overlapping components shift each other's peaks)
%   2. set rateExp=0 on one component (fixed latency) and watch the curves
% in the "rescaled lag" panel of figure 3 stop overlapping
%   3.cfg.ampMode= 'area' keeps the kernel's area instead of its peak under
%   rescaling.
%   4. add a new component type: one more case in componentShape() 

clear; close all; clc

%% --- 1. CONFIG ---
cfg.fs=1000; % Hz
cfg.tWin=[-0.1 1.00]; % s, lag window; kernel is causal (0 before onset)
cfg.tShow=[-0.1 0.7]; % s, x-limits of the time-domain plots
cfg.nfft=2^14; % zero-padded FFT length (freq. resolution = fs/nfft)
cfg.fMin=0.5; % Hz, lower edge of the peak-frequency search
cfg.fMax=20; % Hz, upper edge of the spectrum-plots / peak-frequency search
cfg.peakThr=0.05; % ignore extrema smaller than this fraction of max|h|
cfg.nPk=4; % number of extrema (in time order) to track;

cfg.preset='gamma'; % 'gamma' | 'biexp' | 'dsine'

% Speech rate: the parameters in section 2 are defined at refRate. 
% At any other rate r, each component's delay and time constants are 
% multiplied by (r/refRate)^(-rateExp). (TODO: what this mean?)
cfg.refRate=4.5; % syl/s;
cfg.rateFactors=[2/3 1 3/2]; %>1 = faster speech
cfg.ampMode='peak'; % 'peak': peak amplitude kepts under rescaling (area ~ 1/rate)
                    % 'area': area (DC gain) kept (peak amplitude ~ rate)


%% --- 2. KERNEL DEFINITION ---
% units: seconds, Hz, a.u.
%  each component is a struct with name, type, amp (signed peak), delay,
%  type-specific time constants
% rateExp: how strongly the component's timing follows speech rate:
% 1=fully time-compressed with rate, 0=fixed latency, 0.1=partial;
%  'sweep' selects the parameter explored in section 4 (values are
%  multipliers). TODO: what dis mean?
switch cfg.preset
    case 'gamma' % P1-N1-P2 complex from three gamma ('alpha-like') bumps
        % tuned so the SUMMED kernel peaks near 50, 105, 185 ms.
        % Overlapping components shift and shrink each other's peaks
        % (stand-alone peaks: 55 / 100 / 171ms)
        % so tune against summed-kernel numbers printed below, not the
        % parameters.
        comps{1}=struct('name','P1','type','gamma','amp',0.55,'delay',0.025, ...
            'tau',0.015,'n',2,'rateExp',1);
        comps{1}=struct('name','N1','type','gamma','amp',-1.15,'delay',0.050, ...
            'tau',0.025,'n',2,'rateExp',1);
        comps{1}=struct('name','P2','type','gamma','amp',0.90,'delay',0.125, ...
            'tau',0.023,'n',2,'rateExp',1);
        sweep=struct('comp',2,'field','tau','mult',[0.6 0.8 1 1.25 1.5]);
    
    case 'biexp' % same idea, but literally sums of exponentials
        comps{1}=struct('name','P1','type','biexp','amp', ...
            0.5,'delay',0.035,'tauRise',0.008, 'tauDecay',.032,'rateExp',1);
        comps{2}=struct('name','N1','type','biexp','amp', ...
            -1.1, 'delay',0.055,'tauRise',0.03,'tauDecay',0.060,'rateExp',1);
        comps{3}=struct('name','P2','type','biexp','amp', ...
            1.0,'delay',0.035,'tauRise',0.03,'tauDecay',0.05,'rateExp',1);
        sweep=struct('comp',2,'field','tauDecay','mult',[0.6 0.8 1 1.25 1.5]);
    case 'dsine' % a single damped oscillation ("ringing" evoked response)
        comps{1}=struct('name','ring','type','dsine','amp', 1, ...
            'delay',0.015,'tau',0.150,'freq',6,'rateExp',1);
        sweep=struct('comp',1,'field','freq','mult',[0.6 0.8 1 1.25 1.5]);

    otherwise
        error('Unknown preset %s', cfg.preset);
end

%% --- 3. REFERENCE KERNEL (speech rate = refRate) ---
t=(cfg.tWin(1):1/cfg.fs:cfg.tWin(2))'; % lag axis in s
opts=struct('refRate',cfg.refRate,'ampMode',cfg.ampMode);

[h0, parts0]=kernelFromParams(t, comps, cfg.refRate, opts);
m0=kernelMetrics(t,h0,cfg);

K=numel(comps);
names=cellfun(@(c) c.name, comps, 'UniformOutput',false);
colK=lines(K);

fprintf('\nReference kernel, preset "%s", %.2f syll/s\n ', cfg.preset, cfg.refRate);
printKernel('summed kernel', m0);
for k=1:K
    printKernel(['Component ' names{k}], mernelMetrics(t, parts0(:,k), cfg), false);
end


%% --- LOCAL FUNCTIONS ---
function [h, parts]=kernelFromParams(t,comps, rate, opts)
% sum of components at a given speech rate.
% t: lag axis (s), col vector
% comps: cell array of component structs 
% rate: speech rate (syll/s); opts.refRate: rate at which the parameters
% are defined
% opts.ampMode: 'peak' or 'area' (what is held fixed when kernel is
% rescaled)
% h: summed kernel; parts: [numel(t) x K] individual (weighted) components
% NOTE: not sure I quite follow the functionality toggled by ampMode...
r=rate/opts.refRate;
parts = zeros(numel(t), numel(comps));
for k=1:numel(comps)
    c=comps{k};
    if ~isfield(c, 'rateExp'), c.rateExp=1; end
    s=r^(-c.rateExp); % time-scale factor: s<1 compresses, s>1 stretches
    a=c.amp;
    if strcmpi(opts.ampMode,'area'), a=a/s; end % keep area (DC gain) constant
    parts(:,k)=a*componentShape(t,c,s);
end
h=sum(parts,2);
end

function w=componentShape(t,c,s)
% Unit-peak shape of one causal component. All times are multiplied by s
% (and/or frequencies divided by s) which is exactly a time-axis rescaling.
tp=max(t-c.delay*s, 0); % time since (scaled) onset, 0 before onset
switch lower(c.type)
    case 'gamma'
        % generalized alpha function
        tau=c.tau*s;
        w = (tp ./ (c.n * tau)).^c.n .* exp(c.n - tp ./ tau);
    case 'biexp'
        % difference of two exponentials
        assert(c.tauDecay>c.tauRise, 'biexp needs tauDecay > tauRise to make sense');
        tr=c.taurise * s;
        td=c.tauDeca * s;
        tpk= tr*td/(td - tr) * log(td/tr); % analytic peak time
        w = (exp(-tp ./ td) - exp(-tp ./ tr)) ./ (exp(-tpk/td) - exp(-tpk/tr));
    case 'dsine'
        % damped oscillation
        tau=c.tau * s;
        om=2*pi*c.freq / s;
        tpk=atan(om*tau)/om; % time of first (largest) lobe
        w = exp(-tp ./ tau) .* sin(om*tp) / (exp(tpk/tau) * sin(om*tpk));
    otherwise
        error('Unknown component type "%s"', c.type);
    
end
end

function m=kernelMetrics(t,h,cfg)
% Peak latencies / amplitudes (local mamxima AND minima) and the spectral
% peak frequency.
[tPk, aPk]=findExtrema(t,h,cfg.peakThr);
m.tPk = nan(1, cfg.nPk);
m.aPk = nan(1,cfg.nPk);

n = min(numel(tPk), cfg.nPk);

m.tPk(1:n)=tPk(1:n);
m.aPk(1:n)=aPk(1:n);
[m.f, m.P]=powerSpectrum(h,cfg.fs, cfg.nfft);
band=m.f >= cfg.fMin * m.f <= cfg.fMax;
fb= m.f(band);
[~, iMax]=max(m.P(band));
m.fPeak=fb(iMax);
end

function [tPk, aPk]=findExtrema(t,h,relThr)
end