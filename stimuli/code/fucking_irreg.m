%% load peakrate from file to save time
clear
global boxdir_mine
pr_file=sprintf('%s/stimuli/peakRate/og.mat',boxdir_mine);
sil_tol=0.75;
p_t=0.105;
w_t=2.026;
wf_length=2822400; % should be the same for all stimuli
load(pr_file,"peakRate","fs")

% arrange intervals into column vectors
s_intervals_og=[];
s_intervals_warped=[];
for n_stim=1:size(peakRate,2)
    Ifrom=peakRate(n_stim).times;
    p=peakRate(n_stim).prominence;
    w=peakRate(n_stim).peakwidth;
    % threshold peaks
    Ifrom=Ifrom(p>p_t&w>w_t);
    
    
    %% get segments
    
    seg=[[1; find(diff(Ifrom)>sil_tol)+1] [find(diff(Ifrom)>sil_tol); length(Ifrom)]];
    %inter-segment interval
    ISI=0;
    
    %%
    n_segs=size(seg,1);
    f_center=4;
    interval_ceil_out=0.75;
    for ss=1:n_segs 
        % % get intervals for current segment
        IPI0_seg=diff(Ifrom(seg(ss,1):seg(ss,2)));
        % get original segment duration for post-warp normalization
        seg_dur_0=sum(IPI0_seg);
        IPI1_seg=nan(size(IPI0_seg));
        peakRate_cutoff=8; % in Hz, rate which is considered too fast to count as new syllable from input distribution
        filter_fast_intervals=true;
        if filter_fast_intervals
            too_fast=(1./IPI0_seg)>peakRate_cutoff;
        else
            too_fast=(1./IPI0_seg)>inf;
        end
        % leave overly fast intervals unchanged
        IPI1_seg(too_fast)=IPI0_seg(too_fast);
        slow=1./IPI0_seg<f_center;
        fast=(1./IPI0_seg>f_center)&~too_fast;
        % RULE 10
        %irreg
        % generate random rates from uniform distribution across
        % range of possible values 
        min_stretch_rate=1./sil_tol;
        max_stretch_rate=peakRate_cutoff;
        % % leave overly fast intervals unchanged
        % IPI1_seg(too_fast)=IPI0_seg(too_fast);
        IPI1_seg(~too_fast)=1./(min_stretch_rate+(max_stretch_rate-min_stretch_rate).*rand(sum(~too_fast),1));
    
        if ss>1
            ISI=Ifrom(seg(ss,1))-Ifrom(seg(ss,1)-1);
            start_t=Ito(end)+ISI;
        else
            %won't ISI just be zero for first segment....?
            start_t=Ifrom(seg(ss,1))+ISI;
        end
        
        seg_dur_1=sum(IPI1_seg);
        normalize_segments=false;
        if normalize_segments
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
    end
    % convert to indices
    s = round([Ifrom Ito]*fs);
    % pad indices with correct start/end times
    s = [ones(1,2); s; ones(1,2)*wf_length];
    % fix last to match length of silence in original recording's ending
    end_sil = diff(s(:,1)); end_sil = end_sil(end);
    s(end,2) = s(end-1,2)+end_sil;
    clear Ito Ifrom
    % stack into vectors for plotting histograms

    % remove start/end anchorpoints
    s([1,end],:)=[];
    s_intervals=diff(s)./fs;
    s_intervals_og=cat(1,s_intervals_og,s_intervals(:,1));
    s_intervals_warped=cat(1,s_intervals_warped,s_intervals(:,2));
end

    

% filter out long pauses
s_intervals_og(s_intervals_og>sil_tol)=[];
s_intervals_warped(s_intervals_warped>sil_tol)=[];