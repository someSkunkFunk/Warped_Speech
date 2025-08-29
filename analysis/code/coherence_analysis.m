% coherence analysis

% note: starting with our already preprocessed data and envelopes despite
% having followed a slightly different pipeline than Oganian et al.
% eeg data is bandpassed between 1-15 Hz and then downsampled here
% rather than simply notch-filtered for line noise then downsampled
% it may be interesting to look at results without the filtering though as
% we know filtering itself affects the timing of information and that will
% directly impact coherence
clearvars -except boxdir_mine boxdir_lab


% do sinlge condition first, then try looking at breakdown per condition
separate_conditions=false;
subj=2;
preprocess_config=config_preprocess(subj);
trf_config=config_trf(subj,~separate_conditions,preprocess_config);
load(preprocess_config.preprocessed_eeg_path,"preprocessed_eeg");
stim=load_stim_cell(preprocess_config,preprocessed_eeg);
resp=preprocessed_eeg.resp;
fs=preprocessed_eeg.fs;
n_electrodes=size(resp{1},2);
n_trials=length(stim);
%% compute coherence using mscohere

%todo: group by trial condition + average across subjects?
%preallocate mscac... note that I'm not sure how to determine the number of
%frequencies but I think it depends on nfft and the signal lengths... below
%is based on default settings of mscohere
mscac=nan(n_trials,1025,n_electrodes);
for tt=1:n_trials
    [mscac(tt,:,:), freqs]=mscohere(stim{1},resp{1},[],[],[],fs);
end
%%
%note: oganian & chang averaged across sensors... sensible to do here as
%well?
figure, plot(freqs, squeeze(mean(mscac,[1,3])));
xlabel('frequency (hz)')
ylabel('Magnitue-Squared Coherence')
title(sprintf('subj %d Mean mscoh (across all trials/electrodes)',subj))

% do we

%% my custom routine which did not pan out:
%% test to see if we can get tf representation of a single stimulus:

%PROBLEM: getting a bunch of nans... not all of them, but enough that
%result is not usable...
[cfs,env_tf,eeg_tf]=get_tf(stim{1},resp{1},fs);
%% compute cac based on oganian & chang's description
cac=get_cac(env_tf,eeg_tf);
% todo: iterate over all trials and save so we don't have to recompute each
% time

function cac=get_cac(env_tf,eeg_tf)
%note: I don't think this depends on the frequencies yet?
    ph_eeg=angle(hilbert(eeg_tf));
    ph_env=angle(hilbert(env_tf));

    T=length(env_tf);
    cac=(1/T).*abs(sum(exp(1i*(ph_eeg-ph_env)),2));
end

function [cfs,env_tf,eeg_tf]=get_tf(env,eeg,fs)
    % can't figure out how to design filters like oganian & chang...
    % perhaps need to use something besides a butter...?

    % note: operates on a single trial at a time for the time being because
    % remembering how to stack everything across three different duration
    % trials was breaking my brain and I just want a plot already but I'm
    % sure this could be vectorized and run much faster

    % determine modulation filterbank center frequencies
    %oganian et al used 0.1-octave-spaced betwen 0.67-9 Hz, but our eeg is
    %bandpassed between 1 and 15 Hz and since bandwidths should be 
    % +-0.5 octaves, we'll used 1+.5 octaves to 15-0.5 octaves as our cf
    % range
    bw_octave=0.5; % half-octave bandwidth
    cf_oct_spacing=0.1; % filters will overlap significantly
    min_f=1; % min freq in our eeg data - subject to change
    max_f=15; % max freq in eeg data
    oct_start=log2(min_f*2^.5);
    oct_end=log2(max_f*2.-5);

    
    % end up with center freqs spaced 0.1 octave apart with .5 octave
    % padding within our min/max freqs
    cfs=2.^(oct_start:cf_oct_spacing:oct_end);

    n_bands=numel(cfs);
    [n_samples,n_electrodes]=size(eeg);
    env_tf=nan(n_bands,n_samples);
    eeg_tf=nan(n_bands,n_samples,n_electrodes);
    for ff=1:n_bands
        cf=cfs(ff);
        f_low=cf*2^(-bw_octave);
        f_high=cf*2^(bw_octave);

        % they also designed filter order to allow maximum 3dB passband ripple
        % normalize passabnd edges to Nyquist for buttord
        Wp=[f_low f_high]/(fs/2);
        % and 24 dB min stopband attenuation
        % -> need to define stopband, slightly outside of passband edges
        sb_edge_octave=0.1; % in octave space
        f_stop_low=cf*2^(-bw_octave-sb_edge_octave);
        f_stop_high=cf*2^(bw_octave+sb_edge_octave);
        Ws=[f_stop_low f_stop_high]/(fs/2);
        % passband ripple in dB
        Rp=3;
        % stopband attenuation in dB
        Rs=24;
        
        [n_ord,W_natural]=buttord(Wp,Ws,Rp,Rs);
        [b,a]=butter(n_ord,W_natural,'bandpass');
        % [sos,g]=butter(n_ord,W_natural,'bandpass','sos');
        %note: filtfilt is giving nans when using a,b - using sos is
        %producing an error i dont understand... what if we tried
        %filtfilthd?
        % Hd=dfitl.df1(b,a);

        env_tf(ff,:)=filtfilthd(b,a,env);
        eeg_tf(ff,:,:)=filtfilthd(b,a,eeg);
        
        % env_tf(ff,:)=filtfilt(sos,g,env);
        % eeg_tf(ff,:,:)=filtfilt(sos,g,eeg);
    end
    


    
end