################################################################################
### Hans Rutger Bosker, Radboud University
### HansRutger.Bosker@donders.ru.nl
### Date: 11 November 2024, run in Python 3.11.5 on Windows 11
### License: CC BY 4.0
################################################################################

######################################
# IMPORTING LIBRARIES
######################################

import requests # to send HTTP requests 
import json # for parsing JSON data
import os # for changing the working directory and removing files
import xml.etree.ElementTree as ET # to parse the XML response from webmaus



######################################
# SET PARAMETERS
######################################

# Set the working directory to the location of the script file
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# WebMAUSBasic parameters
webmaus_url = f"https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSBasic"
webmaus_language = 'eng-US'

# runPho2Syl parameters
runpho2syl_url = f"https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runPho2Syl"
runpho2syl_input_tier = "MAU"
runpho2syl_wsync = "yes"
runpho2syl_outsym = "sampa"
runpho2syl_output_format = "tg"  # Output format: TextGrid

# Define audio file extension (e.g., wav or mp3)
audio_extension = "wav"



######################################
# FINDING FILE PAIRS AND PROCESSING
######################################

# Collect all text and audio files in the directory
all_files = os.listdir(script_dir)

# Create sets for easy matching
text_files = {os.path.splitext(f)[0] for f in all_files if f.endswith('.txt')}
audio_files = {os.path.splitext(f)[0] for f in all_files if f.endswith(f'.{audio_extension}')}

# Find common items between text files and audio files
items_to_process = text_files.intersection(audio_files)

# Main loop for processing the data
for item in items_to_process:
    input_text_path = f"{item}.txt"
    input_audio_path = f"{item}.{audio_extension}"

    # Add further processing logic for each pair
    print(f"Processing: {input_text_path} and {input_audio_path}")

    # Run WebMAUSBasic on the mp3 and txt pairs
    with open(input_audio_path, 'rb') as audio_file, open(input_text_path, 'r') as text_file:
        # Construct the payload
        files = {
            'SIGNAL': audio_file,
            'TEXT': text_file
        }
        data = {
            'LANGUAGE': webmaus_language,
            'OUTFORMAT': 'bpf'
        }

        # Send the request to the WebMAUSBasic API
        response = requests.post(webmaus_url, files=files, data=data)

        if response.status_code == 200:
            # Parse the XML response
            root = ET.fromstring(response.content)
            download_link = root.find(".//downloadLink").text

            if download_link:
                # Download the BPF file using the download link
                output_filename_webmaus = f"{item}_webmaus.par"
                download_response = requests.get(download_link)

                # Check if the download was successful
                if download_response.status_code == 200:
                    with open(output_filename_webmaus, 'wb') as f:
                        f.write(download_response.content)
                    #print(f"WebMAUS TextGrid file downloaded as {output_filename_webmaus}")
                    
                    # Step 2: Use the .par BPF file as input to runPho2Syl
                    with open(output_filename_webmaus, 'rb') as bpf_file:
                        files = {
                            'i': bpf_file
                        }
                        data = {
                            'wsync': runpho2syl_wsync,
                            'lng': webmaus_language,
                            'tier': runpho2syl_input_tier,
                            'outsym': runpho2syl_outsym,
                            'oform': runpho2syl_output_format
                        }
                        response_pho2syl = requests.post(runpho2syl_url, files=files, data=data)

                        if response_pho2syl.status_code == 200:
                            # Parse the XML response to find download link for runPho2Syl
                            root_pho2syl = ET.fromstring(response_pho2syl.content)
                            download_link_pho2syl = root_pho2syl.find(".//downloadLink").text

                            if download_link_pho2syl:
                                # Download the runPho2Syl output file
                                output_filename_pho2syl = f"{item}.TextGrid"
                                download_response_pho2syl = requests.get(download_link_pho2syl)
                                
                                if download_response_pho2syl.status_code == 200:
                                    with open(output_filename_pho2syl, 'wb') as f:
                                        f.write(download_response_pho2syl.content)
                                    print(f"TextGrid file saved as {output_filename_pho2syl}")
                                else:
                                    print("Failed to download the runPho2Syl output file.")
                            else:
                                print("Download link not found in the runPho2Syl response.")
                        else:
                            print(f"Failed to process the runPho2Syl request. Status code: {response_pho2syl.status_code}")
                        
                    # Remove the intermediate output files containing only partial output
                    os.remove(output_filename_webmaus)
                else:
                    print("Failed to download the WebMAUS BPF file.")
            else:
                print("Download link not found in the response.")
        else:
            print(f"Failed to process the WebMAUS request. Status code: {response.status_code}")

################################################################################
# End of script
################################################################################
