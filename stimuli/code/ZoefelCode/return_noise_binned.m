function [fnoisen,noise] = return_noise_binned (target_spec, spec_binned, sr, datalen, numof_it, numof_bins, numof_freqs, phase_data)

phases = zeros(numof_bins,1);
for t=1:numof_bins
    phases(t)=-pi+(t-1)*2*pi/(numof_bins-1);
end

noise = 2.*rand(datalen, 1)-1;

spec_new = zeros(length(phase_data),numof_freqs);

for t = 1:length(phase_data)
    [~,bin] = min(abs(phase_data(t)-phases));
    diff_vector = target_spec-spec_binned(bin,:);
    
    spec_new(t,:) = target_spec + diff_vector;
end

for n = 1:numof_it
 if n == round(numof_it/4)
     disp('noise construction: 25%')
 elseif n == round(numof_it/2)
     disp('noise construction: 50%')
 elseif n == round(3*numof_it/4)
     disp('noise construction: 75%')
 end
% transfer noise into wavelet domain
[noisew,PERIOD2,SCALE2,COI2,DJ2, PARAMOUT2, K2] = contwt(noise, 1/sr, -1, 0.05, -1, numof_freqs-1, -1, -1);

fnoisen = zeros(size(noisew));
% scalar product: multiply amplitude spectrum for each of time of the noise
% - normalize by amplitude spectrum of the noise
for t = 1:length(noisew)
    fnoisen(:,t) = noisew(:,t) .* spec_new(t,:)' ./ abs(noisew(:,t));
end

noisi = invcwt(fnoisen, 'morlet', SCALE2, PARAMOUT2,K2);

noise = noisi;

end
