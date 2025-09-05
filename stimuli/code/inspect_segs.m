function inspect_segs(wf,fs,Ifrom,seg,env,p_t,w_t,env_onsets)
% inspect_segs(wf,fs,Ifrom,seg,env)
% plot waveform with segments marked
n_segs=size(seg,1);
t_vec=0:1/fs:(length(wf)-1)/fs;

% columns correspond to different lines
seg_starts=repmat(Ifrom(seg(:,1)),1,2)';
seg_ends=repmat(Ifrom(seg(:,2)),1,2)';
pr_times=repmat(Ifrom,1,2)';
%
seg_ys=ones(2,n_segs);
seg_ys(2,:)=-1;
% env derivative needs scaling to be visible - min should already be zero
env_onsets=normalize(env_onsets,'range',[0 max(env)]);
pr_ys=ones(2,length(pr_times));
pr_ys(2,:)=-1;
figure
plot(t_vec,wf)
hold on
plot(t_vec,env,'Color','c')
plot(pr_times,pr_ys,'Color','m')
plot(t_vec(1:end-1),env_onsets,'Color','y')
plot(seg_starts,seg_ys,'Color','g')
plot(seg_ends,seg_ys,'Color','r')

title(sprintf(['Peakrate (magenta) and ' ...
    'segment boundaries (green/red) p_t:%0.3f,w_t:%0.3f'],p_t,w_t))
ylim([min(wf), max(wf)])
xlabel('Time (s)')
hold off
end