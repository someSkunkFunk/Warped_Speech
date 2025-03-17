
% combine modspectra from noise_test conditions into one mat file
clear,clc
modSpectraDir='./modSpectra';
%folder names are not valid field names... maybe consider renaming to valid
%fieldnames?
% realizing maybe we can keep these weird ass names if we use struc indices
% instead of fields and then just make the "field name" a separate field
% all_conditions={'pure_noise','speechy_noise','speechy_noise_lpf8.5e+03',...
%     'speechy_noise_lpf1.4e+03''speechy_noise_lpf1e+03'};

modSpectraFile1=sprintf('%s/lpfsSpeechNoiseDingMS.mat',modSpectraDir);
ms(1)=load(modSpectraFile1,'modSpectra','conditions');
modSpectraFile2=sprintf("%s/lpfSpeechNoiseLowererCutoffOnlyDingMS.mat",modSpectraDir);
ms(2)=load(modSpectraFile2,'modSpectra','conditions');
modSpectraFile3=sprintf("%s/lpfSpeechNoiseLowerCutoffOnlyDingMS.mat",modSpectraDir);
ms(3)=load(modSpectraFile3,'modSpectra','conditions');

% all have same freqs:
load(modSpectraFile3,'freqs');
conditions={};
outFnm=sprintf('%s/speechNoiseTestDingMS.mat',modSpectraDir);
for mm=1:numel(ms)
    fldnmsCurr=fieldnames(ms(mm).modSpectra);
    nMs=numel(fldnmsCurr);
    for ff=1:nMs
        modSpectra.(fldnmsCurr{ff})=ms(mm).modSpectra.(fldnmsCurr{ff});
        conditions=[conditions; ms(mm).conditions(ff)];
    end
end
% conditions(1)=[];
save(outFnm,"modSpectra","conditions","freqs")