function inspect_anchorpoints(wf,wf_warp,fs,s)
% inspect_anchorpoints(wf,wf_warp,fs,s)
% plot waveform with segments marked
n_anch=size(s,1);
t_og=0:1/fs:(length(wf)-1)/fs;
t_warp=0:1/fs:(length(wf_warp)-1)/fs;
%note: would probably be useful to include segments too but for now just
%look at anchorpoints

% s should have the same number of points in both input and output so only
% need one set of these in theory
line_ys=ones(2,n_anch);
line_ys(2,:)=-1;


figure
subplot(2,1,1)
plot(t_og,wf)
hold on
plot(repmat(t_og(s(:,1)),2,1),line_ys,'Color','m')
title('original waveform')
ylim([min(wf) max(wf)])
ylabel('time (s)')

subplot(2,1,2)
plot(t_warp,wf_warp)
hold on
plot(repmat(t_warp(s(:,2)),2,1),line_ys,'Color','m')
title('warped waveform')
ylim([min(wf_warp) max(wf_warp)])
ylabel('time (s)')
end