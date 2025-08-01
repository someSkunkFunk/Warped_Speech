% generate CORRECT chanlocs file...
clc, clear
raw = readtable('../chanlocs_unrotated.txt', 'Delimiter', '\t', 'ReadVariableNames', ...
    false, 'FileType','text');
x = raw.Var1;
y = raw.Var2;
z = raw.Var3;
labels = raw.Var4;

% Create temporary struct
chanlocs = struct('labels', labels, 'X', num2cell(x), 'Y', num2cell(y), 'Z', num2cell(z));

% Convert Cartesian to polar using EEGLAB
chanlocs = convertlocs(chanlocs, 'cart2topo');
save("../chanlocs.mat")