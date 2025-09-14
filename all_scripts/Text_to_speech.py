from gtts import gTTS

#Module imported to play the converted audio
import os



#Text that needs to be converted to audio
text1 = 'Please Speak now'

#Audio language 
language = 'en'

#Passing the text and language to the engine,  
#Here we have marked slow=False. This tells the module that the converted audio should have a high speed 
myobj1 = gTTS(text=text1, lang=language, slow=False)



#Saving the converted audio in a mp3 file named
file1="objdet1.mp3"

myobj1.save("objdet1.mp3")


#Playing the file

os.system("" + file1)
