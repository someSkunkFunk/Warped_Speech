function [freqs,P]=plot_psd(x,fs,timedim,doplot)
% helper function for plotting amplitude power spectrum, assuming real x
if nargin<3||isempty(timedim)
    timedim=1;
end
if nargin<4||isempty(doplot)
    doplot=true;
end
X=fft(x,[],timedim);
ns=size(X,timedim);
freqs=fs/ns*(0:ns/2);

% keep positive freqs only since real
P=abs(X(1:floor(ns/2)+1));
if doplot
    figure
    plot(freqs,P);
    xlabel('frequency, Hz')
    ylabel('|X|')
end
end