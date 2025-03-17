clear,clc
source_fnms={'compressyRules12345DingMS.mat','stretchyRules12345DingMS.mat'};
ms_dir='./modSpectra';
for fl=1:numel(source_fnms)
    fnm=source_fnms{fl};
    flpth=sprintf('%s/%s',ms_dir,fnm);
    data=load(flpth);
    for fn=1:numel(data.conditions)
        % extract rule/center_f from condition field name
        tokens=regexp(data.conditions{fn},'^rule(\d+)_(.*?)_(median|uquartile|lquartile)','tokens','once');
        
        rule_n=str2double(tokens{1});
        center_used=tokens{3};
        if contains(fnm,'stretchy')
            reg_n=1;
        elseif contains(fnm,'compressy')
            reg_n=2;
        end
        modSpectra(rule_n,reg_n).(center_used)=data.modSpectra.(data.conditions{fn});
    end
end
out_flpth=sprintf('%s/allStretchyCompressyMS.mat',ms_dir);
save(out_flpth,'modSpectra','source_fnms')