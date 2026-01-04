Here we provide MATLAB code for the construction of speech/noise stimuli used in Zoefel and VanRullen (2015, J Neurosci,
2016 and 2017, NeuroImage).

For an input wav-file, fluctuations in spectral content (with respect to differences across phases of the original 
signal envelope) are reduced, whereas fluctuations in "high-level features" (e.g., phonetic information, or any fine
 structure of a given signal, if the input wav-file is not restricted to speech sounds;
 note however that non-speech sounds have not been tested as input yet) are conserved.

The output signal is saved as MATLAB mat- and wav-file, 
and results are plotted as a comparison of fluctuations in spectral content between the original and constructed signals.

Note that a number of iterations are required for a reduction of fluctuations in spectral content.
 The higher this number, the better the performance of the program, but the slower the processing.
 50 iterations usually result in an acceptable performance, but it might take some time, 
depending on the computational power of the system used for stimulus processing.

 
Please type “help construct_stimuli_without_spec_fluc” in MATLAB for information about input/output variables.