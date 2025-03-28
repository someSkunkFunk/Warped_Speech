% Helper script to manage input data access and output storage on box drive
global user_profile
global boxdir_mine
global boxdir_lab
user_profile=getenv('USERPROFILE');
fprintf('setting boxpaths\n')
%TODO move stuff around so this is easier to access/shorter path
boxdir_mine=sprintf(['%s/Box/my box/LALOR LAB/oscillations project/' ...
    'MATLAB/Warped Speech'],user_profile);

boxdir_lab=sprintf(['%s/Box/Lalor Lab Box/Research Projects/' ...
    'Aaron - Warped Speech'],user_profile);

fprintf('boxdir_mine:\n%s\nboxdir_lab:%s\n',boxdir_mine,boxdir_lab)