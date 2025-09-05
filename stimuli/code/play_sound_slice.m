function play_sound_slice(wf,fs,t_start,t_end)
% play_sound_slice(wf,fs,t_start,t_end)
% helper to listen to snippet of waveform by specifying time
t_vec=0:1/fs:(length(wf)-1)/fs;
wf_slice=wf(t_vec>t_start&t_vec<t_end);
soundsc(wf_slice,fs)

end