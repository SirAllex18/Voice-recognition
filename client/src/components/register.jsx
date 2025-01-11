import React, { useState } from "react";
import { Button, CircularProgress, Typography } from "@mui/material";

const VoiceRecorder = () => {
  const [recording, setRecording] = useState(false);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [errorMessage, setErrorMessage] = useState(null);

  let mediaRecorder;
  let audioChunks = [];

  const startRecording = () => {
    setRecording(true);
    setErrorMessage(null);

    navigator.mediaDevices
      .getUserMedia({ audio: true })
      .then((stream) => {
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();

        mediaRecorder.ondataavailable = (event) => {
          audioChunks.push(event.data);
        };

        mediaRecorder.onstop = () => {
          const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
          const audioUrl = URL.createObjectURL(audioBlob);
          setAudioUrl(audioUrl);
          audioChunks = [];

          // Save audio blob as .wav file
          const audioFile = new File([audioBlob], "recording.wav", {
            type: "audio/wav",
          });

          uploadAudio(audioFile);
        };

        // Stop recording after 3-5 seconds
        setTimeout(() => {
          mediaRecorder.stop();
          setRecording(false);
        }, 5000); // Adjust duration here (3000ms = 3 seconds)
      })
      .catch((error) => {
        setErrorMessage("Error accessing microphone: " + error.message);
        setRecording(false);
      });
  };

  const uploadAudio = (audioFile) => {
    setIsUploading(true);
    const formData = new FormData();
    formData.append("audio", audioFile);

    fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Upload successful:", data);
        setIsUploading(false);
      })
      .catch((error) => {
        console.error("Error uploading file:", error);
        setErrorMessage("Failed to upload audio.");
        setIsUploading(false);
      });
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <Typography variant="h4">Voice Recorder</Typography>
      <Button
        variant="contained"
        color="primary"
        onClick={startRecording}
        disabled={recording}
        style={{ marginTop: "20px" }}
      >
        {recording ? "Recording..." : "Start Recording"}
      </Button>
      {recording && <CircularProgress style={{ marginTop: "20px" }} />}
      {audioUrl && (
        <audio
          controls
          src={audioUrl}
          style={{ display: "block", marginTop: "20px" }}
        />
      )}
      {isUploading && <Typography>Uploading...</Typography>}
      {errorMessage && <Typography color="error">{errorMessage}</Typography>}
    </div>
  );
};

export default VoiceRecorder;
