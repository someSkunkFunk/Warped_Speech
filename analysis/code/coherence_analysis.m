% coherence analysis

% note: starting with our already preprocessed data and envelopes despite
% having followed a slightly different pipeline than Oganian et al.
% eeg data is bandpassed between 1-15 Hz and then downsampled here
% rather than simply notch-filtered for line noise then downsampled
% it may be interesting to look at results without the filtering though as
% we know filtering itself affects the timing of information and that will
% directly impact coherence
clearvars -except boxdir_mine boxdir_lab


conditions={'fast','og','slow'};
cond_durs=64.*[2/3,1,3/2];
show_psds=false;
psd_config=[]; %empty for defaults
show_mscohere=false;
mscac_config=[];
show_cac=true;
overwrite_cac=false;
inspect_tf_filters=false;
tf_config=[];
subjs=[2:7,9:22];
% note: computing individual cacs takes a lot of time and memory, so only
% run show_savg_cac=true if the files for each subject exist already, not
% when computing
show_savg_cac=true;
for ss=1:numel(subjs)
    subj=subjs(ss);
    preprocess_config=config_preprocess(subj);
    load(preprocess_config.preprocessed_eeg_path,"preprocessed_eeg");
    %TODO: add preprocess_config to all preprocessed_eeg files for
    %convenience...
    %however, I now see the issue of why I avoided this previously...
    %boxdirs will depend on what machine you run this from, so maybe just
    %stick to this method for now
    preprocessed_eeg.preprocess_config=preprocess_config;
    
    stim=load_stim_cell(preprocess_config,preprocessed_eeg);
    % resp=preprocessed_eeg.resp;
    % fs=preprocessed_eeg.fs;

    if show_psds
        [P_envs, P_eeg, psd_freqs,psd_config]=psds_wrapper(preprocessed_eeg,psd_config);
        %% plot avg psds for stimuli
        %note: maybe adding a 1/f corrective factor (.*sqrt(repmat(psd_freqs',3,1)))
        % would accentuate the coherence pattern... but not sure how to factor that
        % into the mscoherence calculation...

        %note that envelope psds definitely show correspondence with speech
        %rate (albeit hard to see), but eeg doesnt
        %todo: wrap these plots in a function
        stim_psd_xlims=[0 10]; % 
        figure
        plot(psd_freqs,pow2db(P_envs))
        legend(conditions)
        xlim(stim_psd_xlims)
        xlabel('frequency (Hz)')
        ylabel('psd (dB/Hz)')
        title('envelope psds (aka true mod spectra)')
        %% plot avg psds for eeg
        stim_psd_xlims=[0 10]; % 
        figure
        plot(psd_freqs,pow2db(P_eeg))
        legend(conditions)
        xlim(stim_psd_xlims)
        xlabel('frequency (Hz)')
        ylabel('psd (dB/Hz)')
        title(sprintf('subj %d eeg psds',subj))
        %TODO: add plot of cpsd between the two to confirm mscohere
        %result...
    end
    %% compute coherence using mscohere
    if show_mscohere
        
        [mscac,msc_freqs,mscac_config]=mscac_wrapper(preprocessed_eeg,stim,mscac_config);
        %todo: verify that msc_freqs and psd_freqs are interchangible...
        %% plot mscohere result
        %note: oganian & chang averaged across sensors... sensible to do here as
        %well?
        figure, plot(msc_freqs, mscac);
        xlabel('frequency (hz)')
        ylabel('Magnitue-Squared Coherence (a.u)')
        legend(conditions)
        title(sprintf('subj %d Mean mscoh (across all trials/electrodes)',subj))
    end
    % seems to me like there is basically no coherence... how do we actually
    % determine what the noise level for mscoherence is though?
    if show_cac
        tf_config=config_tf(tf_config,preprocessed_eeg.fs);
        if inspect_tf_filters
            show_tf_filters(tf_config)
        end
        %% use oganian & chang coherence metric
        %todo: load precomputed
        global boxdir_mine
        tf_dir=fullfile(boxdir_mine,'analysis','cac');
        cac_fpth=fullfile(tf_dir,sprintf('warped_speech_s%02d.mat',subj));
        if (~exist(cac_fpth,"file")||overwrite_cac)&&~show_savg_cac
            cacs=cacs_wrapper(tf_config,conditions);
            %% save cac result
            fprintf('saving cacs for subj %d...\n',subj)
            save(cac_fpth,"cacs","tf_config","preprocess_config","trf_config")  
            disp('saved.')
        else
            %load
            fprintf('loading existing cacs for subj %d...\n',subj)
            load(cac_fpth)
            if show_savg_cac
                % aggregate all cacs into one massive array
                if ss==1 %preallocate on first subj
                    savg_cacs=nan(numel(subjs),length(conditions),length(tf_config.cfs));
                end
                savg_cacs(ss,:,:)=cacs;
            end
        end
        %% plot cac
        figure
        plot(repmat(tf_config.cfs,length(conditions),1)',cacs');
        legend(conditions);
        xlabel('frequency (hz)')
        ylabel('CAC')
        title(sprintf('CAC subj %d',subj))
        
    end
end
%% TODO: plot across-subjs avg cac... + handle mismatch case...?
if show_savg_cacs
    savg_cacs=squeeze(mean(savg_cacs,1));
    figure
    plot(repmat(tf_config.cfs,length(conditions),1)', savg_cacs');
    legend(conditions)
    xlabel('frequency (hz)')
    ylabel('CAC')
    title('CAC averaged across subjects')
end
%% TODO: use trfs to predict eeg and get that cac also or something idk match/mismatch...?
% probably should do mismatch cacs just to show how mismatching the
% data conditions affects the cac
%% helpers
function cacs=cacs_wrapper(tf_config,conditions)
%todo: add var for match vs mismatched - then adapt code to handle mm...
% also test that it runs correctly in m case?
% also... how to handle differing amounts of data in mm case?
% is there a smarter way to handle that in m case also...?
%preallocate
cacs=nan(length(conditions),numel(tf_config.cfs));
for cc=1:length(conditions)
    fprintf('getting %s cac...\n',conditions{cc})
    m_=preprocessed_eeg.cond==cc;
    stim_=cat(2,stim{m_});
    n_samples_=size(stim_,1);
    % expand stim mat to match eeg mat by repeating each stim for each
    % electrode
    stim_=repelem(stim_,1,n_electrodes);
    resp_=cat(2,resp{m_});
    % get tf representation using filters defined in oganian & chang
    % note: we probably want to save these for future reference but
    % currently they're just overwritten on each loop iteration
    [env_tf_,eeg_tf_]=get_tf(stim_,resp_,tf_config);
    % compute cac based on oganian & chang's description
    switch conditions{cc}
        case 'slow'
            % slow has too many samples to do in one go... split and average cac...
            n_splits_=3;
            % untangle time-dimension for clarity
            env_tf_=reshape(env_tf_,numel(tf_config.cfs),n_samples_,[]);
            eeg_tf_=reshape(eeg_tf_,numel(tf_config.cfs),n_samples_,[]);
            cacs_=nan(n_splits_,numel(tf_config.cfs));
            
            idx_splits_=reshape(1:n_samples_,[],n_splits_)';
            for nn=1:n_splits_
                split_env_tf_=reshape(env_tf_(:,idx_splits_(nn,:),:),numel(tf_config.cfs),[]);
                split_eeg_tf_=reshape(eeg_tf_(:,idx_splits_(nn,:),:),numel(tf_config.cfs),[]);
                cacs_(nn,:)=get_cac(split_env_tf_,split_eeg_tf_);
                clear nn split_env_tf_ split_eeg_tf_
            end
            % note: if we are keeping env_tf/eeg_tf for future reference we
            % should re-flatten the third dimension out again here...
            cacs(cc,:)=mean(cacs_,1);
            clear n_splits_ nn
        case {'fast','og'}
            % technically oganian & chang used the time-split averaging to
            % accounnt for different amounts of data across conditions...
            % maybe should do that here too?
            cacs(cc,:)=get_cac(env_tf_,eeg_tf_);
        otherwise
            error('idk that condition bro')
    end
    % note that averaging across trials/electrodes is implicit in how we
    % defined get_cac
    clear m_ stim_ resp_ cc env_tf_ eeg_tf_ n_samples_
end
end

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
function show_tf_filters(tf_config)
% check that all our filters a designed in a stable fasion
% note couldnt figure oout how to limit the frequanecy axis to zoom in on
% the freqeuncies that matter but the good news is the filters looked
% stable
    n_cac_bands=numel(tf_config.cfs);
    if inspect_filters
        for ff=1:n_cac_bands
            cf=tf_config.cfs(ff);
            [sos,~]=design_bandpass(cf,bw_octave,tf_config.fs);
            figure, freqz(sos,1000,tf_config.fs)
            sgtitle(sprintf('%0.2f Hz cf',cf)) 
        end    
    end
end

function tf_config=config_tf(tf_config,fs)
% define filter cfsbased onoganian & chang vals except different freq range
% bw_octave=0.5; % half-octave bandwidth
% cf_oct_spacing=0.1; % filters will overlap significantly... by design?
% min_f=1; % min freq in our eeg data - subject to change
% max_f=15; % max freq in eeg data
tf_config.fs=fs;
defaults=struct( ...
'bw_octave',0.5,...
'cf_oct_spacing',0.1,...
'min_f',1, ...
'max_f',15 ...
);
fldnms=fieldnames(defaults);
for ff=1:numel(fldnms)
    fldnm=fldnms{ff};
    if ~isfield(tf_config,fldnm)||isempty(tf_config.(fldnm))
        tf_config.(fldnm)=defaults.(fldnm);
    end
end
tf_config.cfs=cut_octave_fbands(tf_config);
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
function [mscac,msc_freqs,psd_config]=mscac_wrapper(preprocessed_eeg,stim,psd_config)
    psd_config=check_psd_config(psd_config,preprocessed_eeg.fs);
    %unpack psd_config vars
    nfft=psd_config.nfft;
    fft_seg_len=psd_config.fft_seg_len;

    cond_nums=unique(preprocessed_eeg.cond)';
    if ~isequal(cond_nums,1:3)
        disp('warning: check that conditions make sense: unique(preprocessed_eeg.cond)=')
        disp(cond_nums)
    end
    mscac=nan(length(cond_nums),nfft/2+1);
    n_electrodes=preprocessed_eeg.preprocess_config.nchan;
    for cc=1:length(cond_nums)
        m_=preprocessed_eeg.cond==cc;
        stim_=cat(2,stim{m_});
        % expand stim mat to match eeg mat by repeating each stim for each
        % electrode
        stim_=repelem(stim_,1,n_electrodes);
        resp_=cat(2,preprocessed_eeg.resp{m_});
        [mscac_, msc_freqs]=mscohere(stim_,resp_, ...
            fft_seg_len,[],nfft,preprocessed_eeg.fs);
        % average out trials/electrodes
        mscac(cc,:)=mean(mscac_,2);
        clear mscac_ stim_ resp_ cc
    end
end
function [P_envs,P_eeg,psd_freqs,psd_config]=psds_wrapper(preprocessed_eeg,psd_config)
    psd_config=check_psd_config(psd_config,preprocessed_eeg.fs);
    %unpack config params
    nfft=psd_config.nfft;
    fft_seg_len=psd_config.fft_seg_len;

    % load all the evelopes
    all_envs=load(preprocessed_eeg.preprocess_config.envelopesFile);
    all_envs=all_envs.env;
    
    cond_nums=unique(preprocessed_eeg.cond)';
    if ~isequal(cond_nums,1:3)
        disp('warning: check that conditions make sense: unique(preprocessed_eeg.cond)=')
        disp(cond_nums)
    end
    %% compute avg psds for stimuli
    
    P_envs=nan(length(cond_nums),nfft/2+1);
    
    for cc=cond_nums
        envs_=cat(2,all_envs{cc,:});
        [envs_,psd_freqs]=pwelch(envs_,fft_seg_len,[],nfft,preprocessed_eeg.fs);
        P_envs(cc,:)=mean(envs_,2); % avg across trials
        clear envs_ cc
    end
    %% compute avg psds for eeg
    P_eeg=nan(length(cond_nums),nfft/2+1);
    for cc=cond_nums
        % trial_idx_=find(preprocessed_eeg.cond==cc);
        resp_=cat(2,preprocessed_eeg.resp{preprocessed_eeg.cond==cc});
        % will give time X n_electrodes*n_trials_cc
        % but mean is linear and we averaging across trials/electrodes anyway
        [resp_,psd_freqs]=pwelch(resp_,fft_seg_len,[],nfft,preprocessed_eeg.fs);
        P_eeg(cc,:)=squeeze(mean(resp_,2));
        
        clear resp_
    end
end
function psd_config=check_psd_config(psd_config,fs)
% set psd configs defaults here - note assumes fft_seg_len is smaller than
% signal it will be applied to (8s for default param - chosen because 
% stimuli are 64s & that divides evenly) 
% also assumes that nfft should be maximal (but it doesn't have to be ...)
defaults=struct( ...
    'fft_seg_dur',8 ...% in s, segment duration for pwelch; gives 1/freq resolution
        );
    fldnms=fieldnames(defaults);
    for ff=1:numel(fldnms)
        fld=fldnms{ff};
        if ~isfield(psd_config,fld)||isempty(psd_config.(fld))
            psd_config.(fld)=defaults.(fld);
        end
    end

if ~isfield(psd_config,'fft_seg_len')
    psd_config.fft_seg_len=round(psd_config.fft_seg_dur*fs)+1;
end
if ~isfield(psd_config,'nfft')
    % note: nfft depends on window length, so by using the same window length
    % across conditions we ensure the same bins despite different durations
    % hence we don't need to specify nfft in pwelch since helpful for
    % preallocating output psd
    psd_config.nfft=2^nextpow2(psd_config.fft_seg_len);
end
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