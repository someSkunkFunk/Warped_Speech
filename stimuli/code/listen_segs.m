function listen_segs(wf,fs,seg,Ifrom)
% listen_segs(wf,fs,seg,Ifrom)
n_segs=length(seg);
buff=1;
t_vec=0:1/fs:(length(wf)-1)/fs;
t_tol=5;
for nn=1:n_segs
    % seg_dur=Ifrom(seg(nn,2))-Ifrom(seg(nn,1));
    seg_start=find(round(t_vec,t_tol)==round(Ifrom(seg(nn,1)),t_tol));
    % listen until start of next seg instead cuz last peakrate gets cut off
    % and I want to hear to make sure it's just one syllable
    % seg_end=find(round(t_vec,t_tol)==round(Ifrom(seg(nn,2)),t_tol));
    if nn<n_segs
        seg_dur=Ifrom(seg(nn+1,1))-Ifrom(seg(nn,1));
        seg_end=find(round(t_vec,t_tol)==round(Ifrom(seg(nn+1,1)),t_tol));
    else
        seg_end=length(wf);
        seg_dur=(seg_end-1)/fs-Ifrom(seg(nn,1));
    end
    fprintf('seg %d of %d (%0.5fs)...\n',nn,n_segs,seg_dur)
    soundsc(wf(seg_start:seg_end),fs)
    pause(seg_dur+buff);
end
end