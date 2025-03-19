% Helper script to manage input data access and output storage on box drive
global user_profile
global input_boxdir
global output_boxdir
user_profile=getenv('USERPROFILE');
%TODO move stuff around so this is easier to access/shorter path
output_boxdir=fprintf(['%s/Box/my box/LALOR LAB/oscillations project/' ...
    'MATLAB/Warped Speech'],user_profile);

input_boxdir=fprintf(['%s/Box/Lalor Lab Box/Research Projects/' ...
    'Aaron - Warped Speech'],user_profile);