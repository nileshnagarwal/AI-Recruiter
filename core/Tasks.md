# TO DO

- [ ] Record a new audio file for each user input and response output. 
- [ ] Save the transcription of the conversation to a text file.
- [ ] Add an exit condition to the conversation based on the response from the LLM.
- [ ] When the response is requested from the LLM, pass the whole conversation history to the LLM. Is it possible to pass a system prompt to the chosen LLM? If so, pass that as well, else append the conversation history to the system prompt, making the system prompt the starting point of the whole prompt passed to the LLM.
- [ ] Record the latency of response between the user stopping his speech and the AI begining to speak.
- [ ] Make the duration of the recording flexible depending on the speech detection of the user. If the user is silent for a while, stop the recording.
- [ ] Enable streaming of audio to the user instead of waiting for the entire response from the LLM, and then converting it to the audio and playing it.
- [ ] Check if whisper is able to detect the language of the user. If not, try to detect it and then transcribe the audio in the detected language. Change the STT model if whisper is not able to detect the language or work in hindi.
- [ ] Implement a barge in funcionality, so that the user can interrupt the LLM response.

- [x] The tet passed on to the LLM is only the text that was transcribed. We need to pass the whole history in the format of <|im_start|>ai agent recruiter<|im_sep|>...<|im_end|>. Also need to include a system prompt with company profile, job description, overall recruitement pipeline and role of the AI recruiter.
- [ ] STT latency alone is the overall latency that we're targeting. Need to experiment with different models to find out what works best.
- [x] The LLM should be informed about the should_exit flag for it to use it correctly.

- [ ] Test hindi transcription. It doesnt seem to be working. 