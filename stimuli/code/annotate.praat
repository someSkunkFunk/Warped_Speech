################################################################################
### Hans Rutger Bosker, Radboud University
### HansRutger.Bosker@ru.nl
### Date: 30 June 2022, run in Praat 6.2.12 on Windows 10
### License: CC BY-NC 4.0
################################################################################

	###>> This script reads a directory containing sound files with pre-existing TextGrids,
	###>>	for instance resulting from a forced aligner (e.g., WebMAUS or EasyAlign).
	###>>	IMPORTANT: Every Sound should have a pre-existing TextGrid file **with the same name**!
	###>>	It opens every Sound + Textgrid combination, presents it to the user for editing,
	###>>	allows the user to enter comments about the annotations, and then saves the
	###>>	edited TextGrid with "_edited" suffix in the subfolder 'edited_textgrids'.
	###>>	User comments are tracked in the file 'annotation_log.txt' in the same subfolder.
	###>>	
	###>>	>>>>> If you **do not** yet have pre-existing TextGrids (i.e., only sound files),	<<<<<
	###>>	>>>>> you can adjust the script to read all .wav files, create new empty TextGrids,	<<<<<
	###>>	>>>>> and present those for editing and saving...									<<<<<
	###>>	>>>>> See the line with "CREATE EMPTY TEXTGRIDS"									<<<<<
	###>>
	###>> This script can be run by multiple users simultaneously, for instance when
	###>>   multiple annotators are working on the same shared folder. It keeps track
	###>>   of what files have already been edited: it only presents TextGrids for editing
	###>>   that do not yet have an "_edited" version pre-existing in the subfolder.
	###>>   This also means that users can close the script or Praat at anytime
	###>>   without losing data. Then, next time someone runs the script, it will
	###>>   start with the files that are 'left over' from the previous run.
	###>>	NOTE: This checking of which files already exist can slow the script down
	###>>	when working with folders with >5000 files...
	###>>
	###>> At present, the script **only** presents pre-existing tiers and intervals
	###>>   for editing (e.g., adding boundaries, dragging boundaries around, etc.).
	###>>   This script can be augmented by automatically adding tiers or intervals
	###>>   before the TextGrid is presented for editing, so users can annotate
	###>>   new tiers/intervals. See the line with "ADD/REMOVE TIERS HERE".

################################################################################
### Variables you will definitely need to customize:
################################################################################

### Where can the Sound and TextGrid files be found?

dir_in$ = "C:\Users\ninet\Box\Lalor Lab Box\Research Projects\Aaron - Warped Speech\stimuli\wrinkle\og\temp_praat_inspection_station"

### Do you want to use 'blinded' objects in Praat to avoid human biases in annotation?
### Default value: "yes"
### Change to "no" if you want to use original object names.

blinded$ = "yes"





################################################################################
### Before we start, let's check whether you've entered sensible
### input for the variables above...
################################################################################

### Let's check if the input directory exists.
### This script will throw an error if the directory doesn't exist
### (i.e., it won't write to a mysterious temp directory).

### First check whether the input directory ends in a backslash (if so, removed)

if right$(dir_in$,1)="/"
	dir_in$ = left$(dir_in$,length(dir_in$)-1)
elsif right$(dir_in$,1)="\"
	dir_in$ = left$(dir_in$,length(dir_in$)-1)
endif

### Then create a temporary txt file in the folder
### and try to write it to the input folder.

### NOTE: The "nocheck" below asks Praat not to complain if the folder
### does *not* exist. We'll manually check whether the saving of this
### temp txt file has succeeded or not further down below.

temp_filename$ = dir_in$ + "/" + "my_temporary_Praat_file.txt"
nocheck writeFileLine: temp_filename$, "This is just to check if the directory exists"

### Can the file be found?

file_exists_yesno = fileReadable(temp_filename$)

if file_exists_yesno = 1
	# if you *could* read that temp txt file,
	# this confirms that the directory is valid.
	# Then you can delete it.
	deleteFile: temp_filename$
else
	# if that file wasn't readable, that means that the directory wasn't valid. 
	printline The folder 'dir_in$' was not found
	exit Your directory doesn't exist. Check spelling. The directory must *already* exist.
endif

## Let's also check whether a subfolder with edited TextGrids already exists,
## for instance when the script has been run and exited before.

temp_filename$ = dir_in$ + "/edited_textgrids/" + "my_temporary_Praat_file.txt"
nocheck writeFileLine: temp_filename$, "This is just to check if the subfolder exists"

### Can the file be found?

subfolderfile_exists_yesno = fileReadable(temp_filename$)

if subfolderfile_exists_yesno = 1
	# if you *could* read that temp txt file,
	# this confirms that the subfolder already exists.
	# Then you can delete it.
	deleteFile: temp_filename$
else
	# if it didn't yet exist, let's create the subfolder
	createFolder: "'dir_in$'/edited_textgrids"
endif





################################################################################
################################################################################
#################################    SCRIPT    #################################
################################################################################
################################################################################

## Let's keep track of who annotated which file. This can be helpful when
## multiple annotators run the same script on the same shared folder.

beginPause: "Please enter your name:"
	text: "annotator", ""
clicked = endPause: "Next", 1

## Now we create a list of TextGrid files in the input directory:

Create Strings as file list: "list_of_files", "'dir_in$'/*.TextGrid"

	#######################################################################################
	## CREATE EMPTY TEXTGRIDS
	########################
	## If you do not yet have pre-existing TextGrids (but a folder with only sound files instead),
	## you can read the sound files in the directory and create empty TextGrids for the user
	## to edit.
	## Adjust this script as follows:
	## - Change *.TextGrid to *.wav in the line above.
	## - Change *.TextGrid to *.wav in the line below starting with "extposition$"
	## - Replace this line: Read from file... 'dir_in$'\'name$'.TextGrid
	##		with this line: To TextGrid: "manual", ""
	#######################################################################################

nfiles = Get number of strings
if nfiles = 0
	exit The directory 'dir_in$' does not contain any TextGrid files.
endif

## By randomizing this file list, it allows for multiple users to simultanously work
## on the same folder without overwriting previous annotations. It also reduces the risk
## of human biases in annotations (e.g., annotator fatigue affecting one condition more
## than another condition).

Randomize

## Now we'll loop through the list and present individual files...

for i from 1 to 'nfiles'
	select Strings list_of_files
	
	fileplusext$ = Get string... 'i'
	extposition = index(fileplusext$, ".TextGrid")
	name$ = left$(fileplusext$, ('extposition'-1))

	outname$ = "'name$'_edited"
	fulloutname$ = "'dir_in$'/edited_textgrids/'outname$'.TextGrid"

	## Let's check if an "_edited" version already exists.
	## The script only presents those files for editing that do not yet have been edited before.
	
	editedfile_exists_yesno = fileReadable(fulloutname$)
	if editedfile_exists_yesno
		do_nothing = 1
	else
		Read from file... 'dir_in$'\'name$'.wav
		if blinded$ = "yes"
			Rename... current_Sound
		endif
		sound_name$ = selected$("Sound")
		## If a filename contains spaces, Praat replaces these spaces with underscores.
		## Example: "file number 1.wav" in a given folder becomes
		##			"file_number_1.wav" in the Praat object window.
		## Therefore, it is important **not** to use a filename variable (here: 'name$')
		## in 'selecting commands' in Praat, like 'select' and 'plus'!
		## Better still: do not use spaces in filenames!
		
		Read from file... 'dir_in$'\'name$'.TextGrid
		if blinded$ = "yes"
			Rename... current_TextGrid
		endif
		tg_name$ = selected$("TextGrid")

		#######################################################################################
		## ADD/REMOVE TIERS HERE
		########################
		## This is where we you could adjust the script to automatically add/remove tiers
		## and/or automatically adjust intervals (setting them to the nearest zero crossings?).
		## Duplicating tiers can be helpful when you want to view original vs. manually edited
		## tiers below/above each other in one and the same edited TextGrid.
		## Example:
		## > Duplicate tier... 1 1 newtier
		## [ARGUMENTS: position_of_tier_to_duplicate position_for_new_tier name_of_new_tier]
		#######################################################################################

		plus Sound 'sound_name$'
		Edit

		beginPause: "Please check and edit this TextGrid."
			comment: "Please check and edit the annotations."
			text: "Comments", ""
		clicked = endPause: "Next", 1
		
		editor TextGrid 'tg_name$'
			Close
		endeditor

		select TextGrid 'tg_name$'
		Write to text file... 'fulloutname$'
		plus Sound 'sound_name$'
		Remove

		appendFileLine: "'dir_in$'/edited_textgrids/annotation_log.txt", annotator$, tab$, name$, tab$, comments$
	endif
endfor

select Strings list_of_files
Remove

################################################################################
# End of script
################################################################################
