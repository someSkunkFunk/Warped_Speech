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
%% compute avg psds for stimuli
% load all the evelopes
if ~exist("all_envs",'var')
    all_envs=load(preprocess_config.envelopesFile);
    all_envs=all_envs.env;
end
%%
conditions={'fast','og','slow'};
cond_durs=64.*[2/3,1,3/2];
seg_dur=10; % in s, segment duration for pwelch; gives 1/freq resolution
seg_len=round(seg_dur*fs)+1;
% note: nfft depends on window length, so by using the same window length
% across conditions we ensure the same bins despite different durations
% hence we don't need to specify nfft in pwelch since helpful for
% preallocating output psd
nfft=2^nextpow2(seg_len);
P_envs=nan(length(conditions),nfft/2+1);
% nfft=seg_len
for cc=1:length(conditions)
    cc_envs_=cat(2,all_envs{cc,:});
    [cc_envs_,psd_freqs]=pwelch(cc_envs_,seg_len,[],nfft,fs);
    P_envs(cc,:)=mean(cc_envs_,2); % avg across trials
    clear cc_envs_
end
%% compute avg psds for eeg
P_eeg=nan(length(conditions),nfft/2+1);
for cc=1:length(conditions)
    % trial_idx_=find(preprocessed_eeg.cond==cc);
    cc_resp_=cat(2,resp{preprocessed_eeg.cond==cc});
    % will give time X n_electrodes*n_trials_cc
    % but mean is linear and we averaging across trials/electrodes anyway
    [cc_resp_,psd_freqs]=pwelch(cc_resp_,seg_len,[],nfft,fs);
    P_eeg(cc,:)=squeeze(mean(cc_resp_,2));
    
    clear cc_resp_
end
%% plot avg psds for stimuli
%note: maybe adding a 1/f corrective factor (.*sqrt(repmat(psd_freqs',3,1)))
% would accentuate the coherence pattern... but not sure how to factor that
% into the mscoherence calculation...
stim_psd_xlims=[0 10]; % 
figure
plot(psd_freqs,P_envs)
legend(conditions)
xlim(stim_psd_xlims)
title('envelope psds (aka true mod spectra)')
%% plot avg psds for eeg
stim_psd_xlims=[0 10]; % 
figure
plot(psd_freqs,P_eeg)
legend(conditions)
xlim(stim_psd_xlims)
title(sprintf('subj %d eeg psds',subj))
%% compute coherence using mscohere

%todo: group by trial condition + average across subjects?
%preallocate mscac... note that I'm not sure how to determine the number of
%frequencies but I think it depends on nfft and the signal lengths... below
%is based on default settings of mscohere
mscac=nan(length(conditions),nfft/2+1);
for cc=1:length(conditions)
    cc_m_=preprocessed_eeg.cond==cc;
    cc_stim_=cat(2,stim{cc_m_});
    % expand stim mat to match eeg mat by repeating each stim for each
    % electrode
    cc_stim_=repelem(cc_stim_,1,n_electrodes);
    cc_resp_=cat(2,resp{cc_m_});
    [cc_mscac_, msc_freqs]=mscohere(cc_stim_,cc_resp_, ...
        seg_len,[],nfft,fs);
    % average out trials/electrodes
    mscac(cc,:)=mean(cc_mscac_,2);
    clear cc_mscac_ cc_stim_ cc_resp_ cc
end

%% plot mscohere result
%note: oganian & chang averaged across sensors... sensible to do here as
%well?
figure, plot(psd_freqs, mscac);
xlabel('frequency (hz)')
ylabel('Magnitue-Squared Coherence')
legend(conditions)
title(sprintf('subj %d Mean mscoh (across all trials/electrodes)',subj))

% seems to me like there is basically no coherence... how do we actually
% determine what the noise level for mscoherence is though?

%% define filter cfs + check that all our filters a designed in a stable fasion
inspect_filters=false;
% note couldnt figure oout how to limit the frequanecy axis to zoom in on
% the freqeuncies that matter but the good news is the filters looked
% stable
tf_config.fs=fs;
tf_config.bw_octave=0.5; % half-octave bandwidth
tf_config.cf_oct_spacing=0.1; % filters will overlap significantly... by design?
tf_config.min_f=1; % min freq in our eeg data - subject to change
tf_config.max_f=15; % max freq in eeg data
tf_config.cfs=cut_octave_fbands(tf_config);
% f_plot=logspace(min_f,max_f,100);
n_cac_bands=numel(cfs);
if inspect_filters
    for ff=1:n_cac_bands
        cf=cfs(ff);
        [sos,g]=design_bandpass(cf,bw_octave,fs);
        figure, freqz(sos,1000,fs)
        sgtitle(sprintf('%0.2f Hz cf',cf)) 
    end    
end
%% use oganian & chang coherence metric
%preallocate
cacs=nan(length(conditions),numel(cfs));
for cc=1:length(conditions)
    fprintf('getting %s cac...\n',conditions{cc})
    cc_m_=preprocessed_eeg.cond==cc;
    cc_stim_=cat(2,stim{cc_m_});
    % expand stim mat to match eeg mat by repeating each stim for each
    % electrode
    cc_stim_=repelem(cc_stim_,1,n_electrodes);
    cc_resp_=cat(2,resp{cc_m_});
    % get tf representation using filters defined in oganian & chang
    [env_tf,eeg_tf]=get_tf(cc_stim_,cc_resp_,tf_config);
    % compute cac based on oganian & chang's description
    cacs(cc,:)=get_cac(env_tf,eeg_tf);
    % note that averaging across trials/electrodes is implicit in how we
    % defined get_cac
    clear cc_m_ cc_stim_ cc_resp_ cc
end
% todo: save result for each subject in analysis folder
%% plot cac
figure
plot(repmat(cfs,length(conditions),1),cacs);
legend(conditions);
xlabel('frequency (hz)')
title(sprintf('CAC subj %s',subj))
% for cc=1:length(conditions)
%     hold on
%     clear cc
% end

%% helpers
function cac=get_cac(env_tf,eeg_tf)
% cac=get_cac(env_tf,eeg_tf)
% env_tf/eeg_tf: [fbands x time*trials*electrodes]
% cac: [fbands x1 ??]
    
    ph_eeg=angle(hilbert(eeg_tf));
    ph_env=angle(hilbert(env_tf));
    % get number of time samples
    [n_bands,T]=size(env_tf);
    %preallocate
    cac=nan(n_bands,1);
    % automatically averages across trials/channels
    cac(:)=(1/T).*abs(sum(exp(1i*(ph_eeg-ph_env)),2));
end

function [env_tf,eeg_tf]=get_tf(env,eeg,tf_config)
    % [env_tf,eeg_tf]=get_tf(env,eeg,tf_config)
    %env: [time x trials*electrodes]
    %eeg: [time x trials*electrodes]
    %env_tf/eeg_tf: [fbands x time*trials*electrodes]
    % note that second dimension is assumed to have a correspondence across
    % the env and eeg samples but can be electrodes, or trials, or both
    % wrapped into one - second dimension is averaged later in cac anyway
    % (across both trials and electrodes) so it doesn't matter much here
    %
    % also note that we're assuming env has it's second dimension expanded
    % to match that of eeg - so this will not work for a single envelope
    % against multiple channels of a single trial
    fs=tf_config.fs;
    bw_octave=tf_config.bw_octave; % half-octave bandwidth
    cf_oct_spacing=tf_config.cf_oct_spacing; % filters will overlap significantly... by design?
    min_f=tf_config.min_f; % min freq in our eeg data - subject to change
    max_f=tf_config.max_f; % max freq in eeg data
    cfs=tf_config.cfs;

    n_bands=numel(cfs);
    [n_samples,n_waveforms]=size(eeg);
    env_tf=nan(n_bands,n_samples*n_waveforms);
    eeg_tf=nan(n_bands,n_samples*n_waveforms);
    for ff=1:n_bands
        cf=cfs(ff);
        [sos,g]=design_bandpass(cf,bw_octave,fs);

        ff_env_tf=filtfilt(sos,g,env);
        ff_eeg_tf=filtfilt(sos,g,eeg);
        % unwrap all fband tf representations into single vector
        env_tf(ff,:)=ff_env_tf(:);
        eeg_tf(ff,:)=ff_eeg_tf(:);
    end    
end
function cfs=cut_octave_fbands(tf_config)
    % determine modulation filterbank center frequencies
    %oganian et al used 0.1-octave-spaced betwen 0.67-9 Hz, but our eeg is
    %bandpassed between 1 and 15 Hz and since bandwidths should be 
    % +-0.5 octaves, we used 1+.5 octaves to 15-0.5 octaves as our cf
    % range
    min_f=tf_config.min_f;
    max_f=tf_config.max_f;
    bw_octave=tf_config.bw_octave;
    cf_oct_spacing=tf_config.cf_oct_spacing;
    oct_start=log2(min_f*2^bw_octave);
    oct_end=log2(max_f*2^-bw_octave);

    % end up with center freqs spaced 0.1 octave apart with .5 octave
    % padding within our min/max freqs
    cfs=2.^(oct_start:cf_oct_spacing:oct_end);
end
function [sos,g]=design_bandpass(cf,bw_octave,fs)
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
        [z,p,k]=butter(n_ord,W_natural,'bandpass');
        [sos,g]=zp2sos(z,p,k);
end

function [psd,freqs]=get_psd(X,fs,method)
% assumes X is 2D matrix that is [time, waveforms] shape

ns=size(X,1);
win_len=round(ns/3);
% win_len=[];
% remove mean
X=detrend(X,'constant');
switch method
    case 'periodogram'
        [psd,freqs]=periodogram(X,hamming(ns),ns,fs);
    case 'pwelch'
        [psd,freqs]=pwelch(X,[],[],[],fs);
end
end