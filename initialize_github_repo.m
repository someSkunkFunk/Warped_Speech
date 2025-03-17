% Change to your project directory
% cd 'Warped Speech/'; already here

% Initialize a new Git repository
system('git init -b main');

% Add all files to the repository
system('git add .');

% Commit the files
system('git commit -m "Initial commit"');

% Add the remote repository (replace with your GitHub repository URL)
remoteRepoURL = 'https://github.com/someSkunkFunk/Warped_Speech.git';
system(['git remote add origin ', remoteRepoURL]);

% Push the changes to the remote repository
system('git push -u origin main');  