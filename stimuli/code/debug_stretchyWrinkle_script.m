%%
% run during execution of stretchyWrinkle to validate warping
% params/peakrate events in a single stimulus
inspect_segs(wf,fs,Ifrom,seg,env,p_t,w_t,diff_env,peakRate,warp_config)

inspect_envelope_derivative(diff_env,peakRate,fs,warp_config)

inspect_anchorpoints(wf,wf_warp,fs,s)