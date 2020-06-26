# TrumpBot: A ChatBot Trained From Donald Trump Transcriptions

## Screenshots

|     Welcome Screen      |      Chat Screen       |     Login Screen      |     Register Screen      |
| :---------------------: | :--------------------: | :-------------------: | :----------------------: |
| ![](images/welcome.png) | ![](images/stored.png) | ![](images/login.png) | ![](images/register.png) |

## TODO

- [x] Scrape transcripts of Donald Trump
- [x] Set up local flask server
- [x] Set up app structure
- [x] Set up networking between app and server
- [x] Train GPT2 model
- [x] Make server return trained model's outputs
- [x] Host server on GCP or some other platform

## Description

- This is an iOS chat app that allows users to chat with a chatbot trained with President Donald Trump transcriptions.
- The corpora was created by web scraping factba.se.
- The chatbot is a pretrained GPT-2 model which is fine-tuned.
- The two corpora (interview and remarks) contain transcripts between Donald Trump
  and a third party. Each line contains the body of text spoken by one party.
- Each line is formatted as <"NAME"> <"SENTIMENT"> <"TEXT">
- The party for each line does not always alternate. Sometimes donald trump speaks
  successively in multiple lines and the same is true for the other party. This is much
  more frequent in the remarks corpus, as it is not an interview but still includes third parties.
  Therefore, the model does not perform as expected.
- The iOS app implements Firebase authentication and Firestore, which allows users to
  load back previously exchanged messages.
