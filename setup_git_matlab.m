% Get the current system PATH
currentPath = getenv('PATH');

% Define the path to the Git executable (replace with your actual Git path)
gitPath = 'C:/Program Files/Git/cmd/git.exe';

% Add the Git path to the system PATH
setenv('PATH', [currentPath ';' gitPath]);

% Verify that Git is accessible from MATLAB
[status, cmdout] = system('git --version');
if status == 0
    disp('Git is successfully configured and accessible from MATLAB.');
    disp(cmdout);
else
    disp('Failed to configure Git. Please check the Git installation and system PATH.');
end 