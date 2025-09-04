function inspect_anchorpoints(wf,wf_warp,fs,s)
% plot waveform with segments marked
n_anch=size(s,1);
t_og=0:1/fs:(length(wf)-1)/fs;
t_warp=0:1/fs:(length(wf_warp)-1)/fs;

% columns correspond to different lines
seg_starts=repmat(Ifrom(seg(:,1)),1,2)';
seg_ends=repmat(Ifrom(seg(:,2)),1,2)';
pr_times=repmat(Ifrom,1,2)';
%
seg_ys=ones(2,n_anch);
seg_ys(2,:)=-1;
% seems times given by findpeaks have some rounding error...?
Ifrom_mask=logical(sum(round(t_og,5)==round(Ifrom,5),1));
pr_ys=repmat(env(Ifrom_mask),1,2)';
pr_ys(2,:)=-1*pr_ys(2,:);
figure
plot(t_og,wf)
hold on
plot(pr_times,pr_ys,'Color','m')
plot(seg_starts,seg_ys,'Color','g')
plot(seg_ends,seg_ys,'Color','r')
plot(t_og,env,'Color','c')
title('Peakrate (magenta) and segment boundaries (green/red)')
ylim([min(wf), max(wf)])
xlabel('Time (s)')
hold off
end