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
resp=preprocessed_eeg.resp;
stim=load_stim_cell(preprocess_config,preprocessed_eeg);
%% test to see if we can get tf representation of a single stimulus:
fs=preprocessed_eeg.fs;
%PROBLEM: getting a bunch of nans... not all of them, but enough that
%result is not usable...
[cfs,env_tf,eeg_tf]=get_tf(stim{1},resp{1},fs);

%%
cac=get_cac(env_tf,eeg_tf);
% todo: iterate over all trials and save so we don't have to recompute each
% time

function cac=get_cac(env_tf,eeg_tf)
%note: I don't think this depends on the frequencies yet?
    ph_eeg=angle(hilbert(eeg_tf));
    ph_env=angle(hilbert(env_tf));

    T=length(env_tf);
    cac=(1/T).*abs(sum(exp(i*(ph_eeg-ph_env)),2));
end

function [cfs,env_tf,eeg_tf]=get_tf(env,eeg,fs)
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
        
        env_tf(ff,:)=filtfilt(b,a,env);
        eeg_tf(ff,:,:)=filtfilt(b,a,eeg);
    end
    


    
end