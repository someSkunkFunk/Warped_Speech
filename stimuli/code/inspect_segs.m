function inspect_segs(wf,fs,Ifrom,seg,env,p_t,w_t,diff_env,peakRate,warp_config)
% inspect_segs(wf,fs,Ifrom,seg,env)
% plot waveform with segments marked
t_tol=5;
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
diff_env=normalize(diff_env,'range',[0 max(env)]);
% [~,pk_idx]=findpeaks(env_onsets);
pk_mask=round(t_vec(1:end-1),t_tol)==round(peakRate.pkTimes,t_tol);
pk_mask=logical(sum(pk_mask,1));
% some peaks still hard to see...
diff_env(pk_mask)=max(env);
pr_ys=ones(2,length(pr_times));
pr_ys(2,:)=-1;
figure
plot(t_vec,wf)
hold on
plot(t_vec,env,'Color','c')
plot(pr_times,pr_ys,'Color','m')
plot(t_vec(1:end),diff_env,'Color','y')
plot(seg_starts,seg_ys,'Color','g')
plot(seg_ends,seg_ys,'Color','r')
legend(sprintf('%s env',warp_config.env_method))

title(sprintf(['Peakrate (magenta) and ' ...
    'segment boundaries (green/red) p_t:%0.4f,w_t:%0.3f'],p_t,w_t))
ylim([min(wf), max(wf)])
xlabel('Time (s)')
hold off
end