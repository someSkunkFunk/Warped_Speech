% Screen 1
Screen('DrawText', win0, 'In this experiment you''ll listen to an audiobook',Lmargin,firstL,cWhite0); nextL = firstL+H;
Screen('DrawText', win0, 'in short excerpts and answer comprehension questions',Lmargin,nextL,cWhite0); nextL = nextL+H;
Screen('DrawText', win0, 'after some of the excerpts. During some segments',Lmargin,nextL,cWhite0); nextL = nextL+H;
Screen('DrawText', win0, 'the speech will sound natural. Other segments will',Lmargin,nextL,cWhite0); nextL = nextL+H;
Screen('DrawText', win0, 'be sped up or slowed down. Finally, the cadence on',Lmargin,nextL,cWhite0); nextL = nextL+H;
Screen('DrawText', win0, 'some segments will be more random or more regular',Lmargin,nextL,cWhite0); nextL = nextL+H;
Screen('DrawText', win0, 'than natural speech. On each trial, do your best to',Lmargin,nextL,cWhite0); nextL = nextL+H;
Screen('DrawText', win0, 'listen and follow the story.',Lmargin,nextL,cWhite0); nextL = nextL+H+H; %LAST LINE 2*H

Screen('DrawText', win0, 'Press any key to continue.',Lmargin,nextL+H,cWhite0);
Screen('Flip',win0);
keytest_unbound;